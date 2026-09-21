"""
Experiment 09 -- vizexp label: viz:ce-pole-count (Section 6.5, Cor. 6.9)

Paper: "Counting the missing poles: MSE vs. cross-entropy on the same
MNIST MLP." Proxy: small TinyMLP classification head on synthetic data.

Verifies Corollary 6.9: MSE gives up to min(n, N*C) non-trivial poles;
softmax cross-entropy gives at most min(n, N*(C-1)) -- exactly N fewer
(one logit-shift null direction per example), because Lambda_CE @ 1_C = 0.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, numerical_jacobian
from pole_dynamics.plotting import savefig


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run():
    rng = set_seed(9)
    C = 3
    N = 8
    mlp = TinyMLP([4, 8, C], activation="tanh", seed=9)
    X = rng.normal(size=(N, 4)) * 0.4
    labels = rng.integers(0, C, size=N)

    theta = mlp.theta0.copy()

    def logits_flat(th):
        return mlp.forward(th, X).ravel()

    J = numerical_jacobian(logits_flat, theta)  # (N*C, n) -- d(logits)/d(theta)
    n = theta.size

    logits = mlp.forward(theta, X)
    p = softmax(logits)

    # MSE Lambda = I_{N*C}
    Lambda_mse = np.eye(N * C)
    # CE Lambda = block-diag(diag(p_i) - p_i p_i^T)
    Lambda_ce = np.zeros((N * C, N * C))
    for i in range(N):
        pi = p[i]
        block = np.diag(pi) - np.outer(pi, pi)
        Lambda_ce[i * C:(i + 1) * C, i * C:(i + 1) * C] = block

    def rank_of_G(Lambda):
        Lh = np.linalg.cholesky(Lambda + 1e-10 * np.eye(N * C)) if np.all(np.linalg.eigvalsh(Lambda) > -1e-9) else None
        # more robust: eigendecomposition sqrt
        w, V = np.linalg.eigh(Lambda)
        w = np.clip(w, 0, None)
        Lsqrt = V @ np.diag(np.sqrt(w)) @ V.T
        Jt = Lsqrt @ J
        sv = np.linalg.svd(Jt, compute_uv=False)
        return sv, np.sum(sv > 1e-8 * (sv[0] + 1e-30))

    sv_mse, r_mse = rank_of_G(Lambda_mse)
    sv_ce, r_ce = rank_of_G(Lambda_ce)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(np.arange(1, len(sv_mse) + 1), sv_mse, "o-", label=f"MSE (rank={r_mse})")
    ax.semilogy(np.arange(1, len(sv_ce) + 1), sv_ce, "s-", label=f"cross-entropy (rank={r_ce})")
    ax.axvline(N * C, color="gray", ls=":", label=f"N*C={N*C}")
    ax.axvline(N * (C - 1), color="k", ls="--", label=f"N*(C-1)={N*(C-1)}")
    ax.set_xlabel("mode index"); ax.set_ylabel("singular value of Lambda^(1/2) J")
    ax.set_title("Corollary 6.9: CE loses N poles to the logit-shift null direction")
    ax.legend(fontsize=8)
    savefig(fig, "exp09_ce_pole_count.png")

    print("Experiment 09 (CE vs MSE pole count):")
    print(f"  n={n}, N={N}, C={C}: min(n,N*C)={min(n, N*C)}, min(n,N*(C-1))={min(n, N*(C-1))}")
    print(f"  measured rank: MSE={r_mse}, CE={r_ce}  (CE should be <= MSE - up to N less)")


if __name__ == "__main__":
    run()
