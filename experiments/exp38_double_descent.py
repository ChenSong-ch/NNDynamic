"""
Experiment 38 -- vizexp label: viz:double-descent (Appendix A.2, Cor. A.2)

Verifies Corollary A.2's conditional mechanism: the generalisation bound
Xi_N/mu (Proposition 10.3) blows up wherever mu=lambda_min,ne0(G) dips.
We use a random-features regression model (Mei & Montanari, 2022's exact
setting cited by the paper) and sweep feature-count/N ratio through the
interpolation threshold, showing lambda_min(X^T X) dips to ~0 there --
the Marchenko-Pastur mechanism the paper's own scope remark points to.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(38)
    N = 60
    widths = np.arange(5, 200, 5)
    mu_vals, test_err = [], []

    x_true = rng.normal(size=20)
    X_raw = rng.normal(size=(N + 200, 20))
    y_all = X_raw @ x_true + rng.normal(size=N + 200) * 0.5
    Xtr, ytr = X_raw[:N], y_all[:N]
    Xte, yte = X_raw[N:], y_all[N:]

    for p in widths:
        Wrf = rng.normal(size=(20, p)) / np.sqrt(20)
        Ztr = np.maximum(Xtr @ Wrf, 0)  # ReLU random features
        Zte = np.maximum(Xte @ Wrf, 0)
        G = Ztr.T @ Ztr / N
        eigs = np.linalg.eigvalsh(G)
        mu = eigs[eigs > 1e-8].min() if np.any(eigs > 1e-8) else 1e-8
        mu_vals.append(mu)
        reg = 1e-6
        beta = np.linalg.solve(Ztr.T @ Ztr + reg * np.eye(p), Ztr.T @ ytr)
        pred = Zte @ beta
        test_err.append(np.mean((pred - yte) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(widths, test_err, "o-")
    axes[0].axvline(N, color="r", ls="--", label="interpolation threshold (p=N)")
    axes[0].set_xlabel("random-feature width p"); axes[0].set_ylabel("test MSE")
    axes[0].set_title("Double-descent test-error curve")
    axes[0].legend()

    axes[1].semilogy(widths, mu_vals, "s-", color="C1")
    axes[1].axvline(N, color="r", ls="--", label="interpolation threshold")
    axes[1].set_xlabel("random-feature width p"); axes[1].set_ylabel("mu = lambda_min,ne0(G)")
    axes[1].set_title("Corollary A.2: mu dips near p=N -> generalisation bound Xi_N/mu spikes")
    axes[1].legend()
    savefig(fig, "exp38_double_descent.png")

    idx_peak_err = np.argmax(test_err)
    idx_min_mu = np.argmin(mu_vals)
    print("Experiment 38 (double descent / smallest non-zero Gauss-Newton eigenvalue):")
    print(f"  test-error peak at width={widths[idx_peak_err]}, mu-minimum at width={widths[idx_min_mu]} "
          f"(interpolation threshold N={N})")


if __name__ == "__main__":
    run()
