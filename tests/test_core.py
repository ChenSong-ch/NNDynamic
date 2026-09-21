"""Smoke tests for pole_dynamics.core against closed-form cases."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pole_dynamics import (
    numerical_jacobian, numerical_hessian, hvp_generic,
    numerical_gradient, lanczos_topk, gd_poles, TinyMLP,
)


def test_jacobian_linear_map():
    A = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    f = lambda x: A @ x
    J = numerical_jacobian(f, np.array([0.3, -0.2]))
    assert np.allclose(J, A, atol=1e-6)


def test_hessian_quadratic():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    f = lambda x: 0.5 * x @ Q @ x
    H = numerical_hessian(f, np.array([0.1, -0.4]))
    assert np.allclose(H, Q, atol=1e-4)


def test_hvp_matches_full_hessian():
    Q = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    f = lambda x: 0.5 * x @ Q @ x
    x0 = np.array([0.2, -0.1, 0.05])
    v = np.array([1.0, 0.0, 0.0])
    hv = hvp_generic(f, x0, v)
    assert np.allclose(hv, Q @ v, atol=1e-3)


def test_gradient_linear_plus_quadratic():
    Q = np.eye(3) * 2.0
    b = np.array([1.0, -2.0, 0.5])
    f = lambda x: 0.5 * x @ Q @ x + b @ x
    x0 = np.array([0.1, 0.2, -0.3])
    g = numerical_gradient(f, x0)
    assert np.allclose(g, Q @ x0 + b, atol=1e-5)


def test_lanczos_topk_matches_eigh():
    rng = np.random.default_rng(0)
    n = 12
    A = rng.normal(size=(n, n))
    Q = A + A.T  # symmetric
    hvp = lambda v: Q @ v
    evals, _ = lanczos_topk(hvp, n, k=4, iters=n, seed=1)
    true_evals = np.sort(np.linalg.eigvalsh(Q))[::-1][: len(evals)]
    approx_evals = np.sort(evals)[::-1]
    # Lanczos with full iters should recover extremal eigenvalues closely
    assert np.abs(approx_evals[0] - true_evals[0]) < 1e-6


def test_gd_poles_formula():
    eigs = np.array([1.0, 2.0, 0.5])
    eta = 0.1
    z = gd_poles(eigs, eta)
    assert np.allclose(z, 1 - eta * eigs)


def test_tiny_mlp_forward_shape():
    mlp = TinyMLP([3, 5, 2], activation="tanh", seed=0)
    X = np.random.default_rng(0).normal(size=(4, 3))
    out = mlp.forward(mlp.theta0, X)
    assert out.shape == (4, 2)
