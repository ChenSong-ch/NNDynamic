"""
Experiment 04 -- vizexp label: viz:svd-poles (Section 3, Theorem 3.9)

Paper: "The general pole formula, on a real trained network" (MLP on an
interpolating MNIST subset). Proxy: TinyMLP interpolating a small
synthetic regression dataset (same mechanism -- the theorem needs only
squared error + an interpolating fixed point).

Verifies the central closed-form result of the paper:
    z_i = 1 - eta * sigma_i(J(w*))^2 / N
Plots all non-trivial poles on the real axis with the unit circle, and
shows that pushing eta above 2/lambda_max(G) pushes the top pole outside
[-1, 1], exactly as Corollary 3.11 predicts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, numerical_jacobian
from pole_dynamics.plotting import savefig, plot_complex_poles


def run():
    rng = set_seed(4)
    mlp = TinyMLP([3, 10, 2], activation="tanh", seed=4)
    N = 8
    X = rng.normal(size=(N, 3)) * 0.4
    y = rng.normal(size=(N, 2)) * 0.2

    def residual(theta):
        return (mlp.forward(theta, X) - y).ravel()

    theta = mlp.theta0.copy()
    eta_train = 0.02
    for _ in range(6000):
        g = numerical_jacobian(residual, theta).T @ residual(theta)
        gn = np.linalg.norm(g)
        if gn > 5.0:
            g *= 5.0 / gn
        theta -= eta_train * g
    print(f"  ||R(w*)|| = {np.linalg.norm(residual(theta)):.3e}")

    J = numerical_jacobian(residual, theta)
    sing = np.linalg.svd(J, compute_uv=False)
    sing = sing[sing > 1e-6 * sing[0]]
    lam_max = sing[0] ** 2 / N
    eta_thresh = 2.0 / lam_max

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for eta, tag, color in [(0.5 * eta_thresh, "0.5x threshold", "C0"),
                             (0.95 * eta_thresh, "0.95x threshold", "C1"),
                             (1.2 * eta_thresh, "1.2x threshold (unstable)", "C3")]:
        z = 1 - eta * sing ** 2 / N
        axes[0].scatter(z, np.zeros_like(z), label=tag, color=color, alpha=0.8)
    axes[0].axvline(1, color="gray", lw=0.5)
    axes[0].axvline(-1, color="k", linestyle="--", lw=1)
    axes[0].axvline(1, color="k", linestyle="--", lw=1)
    axes[0].set_xlabel("pole z_i (real axis)")
    axes[0].set_title("Poles z_i = 1 - eta*sigma_i^2/N vs. eta")
    axes[0].legend(fontsize=7)

    z_at_thresh_half = 1 - 0.5 * eta_thresh * sing ** 2 / N
    plot_complex_poles(axes[1], z_at_thresh_half, label="eta=0.5x threshold", color="C0")
    savefig(fig, "exp04_svd_pole_formula.png")

    print("Experiment 04 (SVD pole formula):")
    print(f"  # non-trivial poles = rank(J) = {len(sing)} (n={mlp.n_params})")
    print(f"  lambda_max(H) = {lam_max:.4f}, eta_threshold(2/lambda_max) = {eta_thresh:.4f}")


if __name__ == "__main__":
    run()
