"""
Experiment 02 -- vizexp label: viz:lyapunov (Section 2.6, Theorems 2.10, 2.12)

Paper: "Both Lyapunov functions, on MNIST logistic regression." We use
scikit-learn's bundled `digits` dataset (8x8 images, 2-class subset) in
place of full MNIST -- same closed-form beta_L, mu as the paper's setup,
no internet download required.

Verifies:
  Theorem 2.10 (loss is a Lyapunov function for eta < 2/beta_L, no convexity
  needed) and Theorem 2.12 (geometric convergence of ||w_k - w*||^2 under
  strong convexity), at eta*beta_L in {0.5, 1.0, 1.5, 1.9, 2.1} x 1/beta_L.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run():
    data = load_digits(n_class=2)
    X = data.data / 16.0
    y = data.target.astype(float)
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # bias
    N, d = X.shape
    lam_reg = 1e-2

    # beta_L: softmax/sigmoid curvature <= 1/4, so beta_L = 1/4*lambda_max(X^T X/N) + lam_reg
    XtX = X.T @ X / N
    lam_max = np.linalg.eigvalsh(XtX).max()
    beta_L = 0.25 * lam_max + lam_reg
    mu = lam_reg

    def loss_and_grad(w):
        z = X @ w
        p = sigmoid(z)
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + 0.5 * lam_reg * (w @ w)
        grad = X.T @ (p - y) / N + lam_reg * w
        return loss, grad

    # long run to approximate w*
    w = np.zeros(d)
    for _ in range(4000):
        _, g = loss_and_grad(w)
        w -= (1.0 / beta_L) * g
    w_star = w.copy()

    ratios = [0.5, 1.0, 1.5, 1.9, 2.1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in ratios:
        eta = r / beta_L
        w = np.zeros(d)
        losses, logV = [], []
        for k in range(300):
            L, g = loss_and_grad(w)
            losses.append(L)
            logV.append(np.log(np.sum((w - w_star) ** 2) + 1e-18))
            w = w - eta * g
        axes[0].plot(losses, label=f"eta*beta_L={r}")
        pred_slope = np.log(max(1 - 2 * eta * mu + eta ** 2 * beta_L ** 2, 1e-12))
        k_arr = np.arange(300)
        axes[1].plot(logV, label=f"eta*beta_L={r}")
        axes[1].plot(k_arr, logV[0] + pred_slope * k_arr, "--", alpha=0.5,
                     color=axes[1].lines[-1].get_color())

    axes[0].set_xlabel("GD step"); axes[0].set_ylabel("L_N(w_k)")
    axes[0].set_title("Theorem 2.10: loss is Lyapunov below eta=2/beta_L")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("GD step"); axes[1].set_ylabel("log ||w_k - w*||^2")
    axes[1].set_title("Theorem 2.12: geometric rate (dashed = predicted slope)")
    axes[1].legend(fontsize=7)
    savefig(fig, "exp02_lyapunov_functions.png")

    print("Experiment 02 (Lyapunov functions):")
    print(f"  beta_L={beta_L:.4f}, mu={mu:.4f}, threshold eta*={2/beta_L:.4f}")


if __name__ == "__main__":
    run()
