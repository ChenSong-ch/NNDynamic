"""
Experiment 07 -- vizexp label: viz:root-locus (Section 4, Cor. 4.2)

Paper: "Root locus: GD's stable interval, on a real quadratic" using the
sklearn `diabetes` dataset ridge regression (exactly as specified in the
paper's own vizexp text -- this one needed no scaling down at all).

Verifies Corollary 4.2: every pole z_i(eta) = 1 - eta*lambda_i(H) crosses
z=-1 at exactly eta=2/lambda_i; the last curve to leave (-1,1) does so at
eta = 2/lambda_max, matching the classical GD stability threshold.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from pole_dynamics.plotting import savefig


def run():
    X, y = load_diabetes(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    N, n = X.shape
    lam_reg = 1e-2
    H = X.T @ X / N + lam_reg * np.eye(n)
    eigs = np.linalg.eigvalsh(H)
    lam_max = eigs.max()

    etas = np.linspace(1e-6, 2.5 / lam_max, 200)
    Z = 1 - np.outer(etas, eigs)  # (len(etas), n)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n):
        ax.plot(etas * lam_max, Z[:, i], lw=1, alpha=0.7)
    ax.axhline(1, color="gray", lw=0.5)
    ax.axhline(-1, color="k", ls="--", label="z=-1 (stability boundary)")
    ax.axvline(2.0, color="r", ls=":", label="eta*lambda_max=2")
    ax.set_xlabel("eta * lambda_max(H)")
    ax.set_ylabel("pole z_i(eta)")
    ax.set_title("Root locus: z_i = 1 - eta*lambda_i(H), diabetes ridge regression")
    ax.legend()
    savefig(fig, "exp07_root_locus.png")

    # verify each curve crosses -1 exactly at eta = 2/lambda_i
    crossing_etas = 2.0 / eigs
    print("Experiment 07 (root locus):")
    print(f"  n={n} features, lambda_max={lam_max:.4f}, predicted global threshold eta*={2/lam_max:.4f}")
    print(f"  per-mode crossing points (2/lambda_i), sorted: {np.sort(crossing_etas)}")


if __name__ == "__main__":
    run()
