"""
Experiment 10 -- vizexp label: viz:multitask (Section 6.6, Theorem 6.11)

Reproduces Script 28's finding directly at small scale: two tasks on
shared parameters, each individually rank-deficient in Gauss-Newton
curvature (blind to different coordinate subsets), whose SUM is full
rank. Verifies Theorem 6.11 / Corollary 6.12: ker(H_combined) =
intersection of the individual ker(G_k) -- auxiliary tasks can only
shrink the null space, never enlarge it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(10)
    n = 6
    # Task 1: blind to coordinates {3,4} (zero columns there)
    J1 = rng.normal(size=(5, n))
    J1[:, [3, 4]] = 0.0
    G1 = J1.T @ J1 / 5

    # Task 2: blind to coordinates {0,1}
    J2 = rng.normal(size=(5, n))
    J2[:, [0, 1]] = 0.0
    G2 = J2.T @ J2 / 5

    def null_dim(G, tol=1e-8):
        eigs = np.linalg.eigvalsh(G)
        return int(np.sum(eigs < tol * max(eigs.max(), 1.0)))

    alphas = np.linspace(0.01, 5, 30)
    null_dims = [null_dim(G1 + a * G2) for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(alphas, null_dims, "o-")
    axes[0].axhline(null_dim(G1), color="C1", ls="--", label=f"dim ker(G1)={null_dim(G1)}")
    axes[0].axhline(null_dim(G2), color="C2", ls="--", label=f"dim ker(G2)={null_dim(G2)}")
    axes[0].set_xlabel("alpha (task-2 weight)")
    axes[0].set_ylabel("dim ker(G1 + alpha*G2)")
    axes[0].set_title("Combined null space <= min(individual null spaces)")
    axes[0].legend()

    eigs1 = np.sort(np.linalg.eigvalsh(G1))
    eigs2 = np.sort(np.linalg.eigvalsh(G2))
    eigs_sum = np.sort(np.linalg.eigvalsh(G1 + G2))
    axes[1].semilogy(eigs1[::-1] + 1e-12, "o-", label="spec(G1)")
    axes[1].semilogy(eigs2[::-1] + 1e-12, "s-", label="spec(G2)")
    axes[1].semilogy(eigs_sum[::-1] + 1e-12, "^-", label="spec(G1+G2)")
    axes[1].set_xlabel("mode index (sorted)"); axes[1].set_ylabel("eigenvalue")
    axes[1].set_title("Multi-task Gauss-Newton additivity (Theorem 6.11)")
    axes[1].legend()
    savefig(fig, "exp10_multitask_null_space.png")

    print("Experiment 10 (multi-task null-space intersection):")
    print(f"  dim ker(G1)={null_dim(G1)}, dim ker(G2)={null_dim(G2)}, "
          f"dim ker(G1+G2)={null_dim(G1+G2)} (paper's Script 28: 2, 2, 0)")


if __name__ == "__main__":
    run()
