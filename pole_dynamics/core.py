"""Core numerical primitives shared by all experiments.

Design choice: rather than depending on a deep-learning framework, every
network here is a small, explicit NumPy function of a *flat* parameter
vector ``theta``.  That lets us compute exact Jacobians/Hessians by
central finite differences (Richardson-quality, step size tuned per
call) without any autodiff dependency -- exactly how the paper's own
Appendix A scripts are described ("plain NumPy/SciPy").  For the sizes
used in this repo (parameter counts of a few dozen to a few thousand)
this is fast and numerically reliable to 1e-6 or better, which is more
than enough to confirm the qualitative and quantitative predictions of
each proposition.
"""
from __future__ import annotations
import numpy as np


def set_seed(seed: int = 0) -> np.random.Generator:
    """Return a seeded numpy Generator (used by every experiment for
    reproducibility)."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# A tiny fully-connected network, flat-parameterised, used as the network
# f(x; w) throughout Part I/II of the paper.
# ---------------------------------------------------------------------------
class TinyMLP:
    """A minimal feed-forward network operating on a flat parameter vector.

    f(x; w) = W_L act( ... act(W_1 x + b_1) ... ) + b_L

    All weights/biases are packed into a single 1-D ``theta`` so that the
    generic finite-difference Jacobian/Hessian helpers below apply to it
    directly -- this mirrors the paper's J(w) = d f / d w object exactly.
    """

    def __init__(self, layer_sizes, activation: str = "tanh", seed: int = 0):
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        rng = set_seed(seed)
        self._shapes = []
        theta = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            W = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_out, n_in))
            b = np.zeros(n_out)
            self._shapes.append((W.shape, b.shape))
            theta.append(W.ravel())
            theta.append(b.ravel())
        self.theta0 = np.concatenate(theta)
        self.n_params = self.theta0.size

    def _act(self, z):
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "linear":
            return z
        raise ValueError(self.activation)

    def unpack(self, theta):
        params = []
        idx = 0
        for (Ws, bs) in self._shapes:
            Wsize = int(np.prod(Ws))
            bsize = int(np.prod(bs))
            W = theta[idx: idx + Wsize].reshape(Ws)
            idx += Wsize
            b = theta[idx: idx + bsize].reshape(bs)
            idx += bsize
            params.append((W, b))
        return params

    def forward(self, theta, X):
        """X: (N, d_in) -> (N, d_out)."""
        params = self.unpack(theta)
        h = X.T  # (d_in, N)
        for li, (W, b) in enumerate(params):
            z = W @ h + b[:, None]
            h = self._act(z) if li < len(params) - 1 else z
        return h.T  # (N, d_out)

    def output_vector(self, theta, X):
        """Flatten f(X; theta) to R^{N * d_out}, i.e. F(w) of the paper."""
        return self.forward(theta, X).ravel()


# ---------------------------------------------------------------------------
# Finite-difference calculus: Jacobians, Hessians, Hessian/Jacobian-vector
# products, all central-difference for O(h^2) accuracy.
# ---------------------------------------------------------------------------
def numerical_jacobian(func, theta, h=1e-5):
    """Jacobian of vector-valued func(theta) w.r.t. theta, shape (m, n)."""
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(func(theta), dtype=float)
    n = theta.size
    m = f0.size
    J = np.zeros((m, n))
    for i in range(n):
        dt = np.zeros(n)
        dt[i] = h
        fp = np.asarray(func(theta + dt), dtype=float)
        fm = np.asarray(func(theta - dt), dtype=float)
        J[:, i] = (fp - fm) / (2 * h)
    return J


def numerical_hessian(scalar_func, theta, h=1e-4):
    """Hessian of scalar_func(theta) via central differences, shape (n, n)."""
    theta = np.asarray(theta, dtype=float)
    n = theta.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dti = np.zeros(n); dti[i] = h
            dtj = np.zeros(n); dtj[j] = h
            if i == j:
                fpp = scalar_func(theta + dti)
                f0 = scalar_func(theta)
                fmm = scalar_func(theta - dti)
                H[i, i] = (fpp - 2 * f0 + fmm) / (h ** 2)
            else:
                fpp = scalar_func(theta + dti + dtj)
                fpm = scalar_func(theta + dti - dtj)
                fmp = scalar_func(theta - dti + dtj)
                fmm = scalar_func(theta - dti - dtj)
                val = (fpp - fpm - fmp + fmm) / (4 * h ** 2)
                H[i, j] = H[j, i] = val
    return H


def finite_diff_hvp(scalar_func, theta, v, h=1e-4):
    """Hessian-vector product H(theta) @ v via a 3-point stencil along v.

    Cheap alternative to forming the full Hessian: O(1) function evals
    per HVP instead of O(n^2).
    """
    v = np.asarray(v, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-15)
    f0 = scalar_func(theta)
    fp = scalar_func(theta + h * v)
    fm = scalar_func(theta - h * v)
    directional_curvature = (fp - 2 * f0 + fm) / (h ** 2)
    return directional_curvature * v  # exact only if v is an eigenvector;
    # for generic v use `hvp_generic` below.


def hvp_generic(scalar_func, theta, v, h=1e-4):
    """True Hessian-vector product H(theta) @ v for a generic direction v,
    via central differences of the gradient along v (O(h^2) accurate),
    using the identity H v = d/dt [grad f(theta + t v)]|_{t=0}."""
    grad = numerical_gradient(scalar_func, theta)
    gp = numerical_gradient(scalar_func, theta + h * v)
    gm = numerical_gradient(scalar_func, theta - h * v)
    return (gp - gm) / (2 * h)


def numerical_gradient(scalar_func, theta, h=1e-6):
    theta = np.asarray(theta, dtype=float)
    n = theta.size
    g = np.zeros(n)
    for i in range(n):
        dt = np.zeros(n)
        dt[i] = h
        g[i] = (scalar_func(theta + dt) - scalar_func(theta - dt)) / (2 * h)
    return g


def lanczos_topk(hvp_fn, n, k=10, iters=40, seed=0):
    """Estimate the top-k eigenvalues (by |lambda|) of a symmetric operator
    given only a matrix-vector product ``hvp_fn(v) -> H v``.

    Used throughout the paper's own experiments ("estimate the top-k
    eigenvalues of G(w_k) via Lanczos on Gauss-Newton-vector products").
    Returns (eigvals, eigvecs) of the small Lanczos tridiagonal matrix,
    sorted by descending |eigenvalue|.
    """
    rng = np.random.default_rng(seed)
    iters = min(iters, n)
    Q = np.zeros((n, iters))
    alpha = np.zeros(iters)
    beta = np.zeros(iters - 1)
    q = rng.normal(size=n)
    q /= np.linalg.norm(q)
    Q[:, 0] = q
    q_prev = np.zeros(n)
    beta_prev = 0.0
    for j in range(iters):
        w = hvp_fn(Q[:, j])
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j] - (beta_prev * q_prev if j > 0 else 0)
        # full reorthogonalisation for numerical stability at this tiny scale
        for i in range(j + 1):
            w -= (Q[:, i] @ w) * Q[:, i]
        nb = np.linalg.norm(w)
        if j < iters - 1:
            beta[j] = nb
            if nb < 1e-12:
                iters = j + 1
                break
            Q[:, j + 1] = w / nb
        q_prev = Q[:, j]
        beta_prev = beta[j] if j < len(beta) else 0.0
    T = np.diag(alpha[:iters]) + np.diag(beta[:iters - 1], 1) + np.diag(beta[:iters - 1], -1)
    evals, evecs_T = np.linalg.eigh(T)
    order = np.argsort(-np.abs(evals))[:k]
    evals = evals[order]
    evecs = Q[:, :iters] @ evecs_T[:, order]
    return evals, evecs


def gd_poles(hessian_eigs, eta):
    """z_i = 1 - eta * lambda_i(H) -- Eq. (poles-general) of the paper."""
    return 1.0 - eta * np.asarray(hessian_eigs)
