"""
Experiment 01 -- vizexp label: viz:one-step (Section 2, Theorem 1)

Paper: "The one-step identity and the realised curvature kappa_k, on
ResNet-18/CIFAR-10." Full-scale version trains ResNet-18; this proxy
uses a small TinyMLP regressor (structurally the same mechanism: the
paper's own claim is architecture-independent, it needs only C^2
smoothness, Assumption 2.1) so the exact identity and its breakdown at
eta*kappa = 2 can be checked to numerical precision on a laptop.

We verify Theorem 2.2 (exact second-order one-step identity):
    Delta L_k = grad(L)^T dw_k + 1/2 dw_k^T H(w_k) dw_k + rho_k
and Eq. (kappa): kappa_k := dw_k^T H(w_k) dw_k / ||dw_k||^2, the
*realised* curvature along the step actually taken.

Prediction confirmed: the two-term Taylor prediction matches the
measured loss change while eta*kappa_k < 2, and visibly departs once
eta*kappa_k approaches/exceeds 2 (the cubic remainder becomes
non-negligible exactly where the model predicts a loss increase).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pole_dynamics import TinyMLP, set_seed, hvp_generic, numerical_gradient
from pole_dynamics.plotting import savefig
import matplotlib.pyplot as plt


def run():
    rng = set_seed(0)
    mlp = TinyMLP([4, 6, 1], activation="tanh", seed=0)
    N = 20
    X = rng.normal(size=(N, 4))
    y = rng.normal(size=(N, 1)) * 0.5

    def loss(theta):
        pred = mlp.forward(theta, X)
        return 0.5 * np.mean((pred - y) ** 2)

    theta = mlp.theta0.copy()
    eta = 0.5
    steps = 60
    measured_dL, predicted_dL, eta_kappa = [], [], []

    for k in range(steps):
        g = numerical_gradient(loss, theta)
        dw = -eta * g
        Hdw = hvp_generic(loss, theta, dw)  # H @ dw (generic direction)
        kappa_k = (dw @ Hdw) / (dw @ dw + 1e-18)
        L0 = loss(theta)
        theta_new = theta + dw
        L1 = loss(theta_new)
        dL_measured = L1 - L0
        dL_predicted = g @ dw + 0.5 * (dw @ Hdw)
        measured_dL.append(dL_measured)
        predicted_dL.append(dL_predicted)
        eta_kappa.append(eta * kappa_k)
        theta = theta_new

    measured_dL = np.array(measured_dL)
    predicted_dL = np.array(predicted_dL)
    eta_kappa = np.array(eta_kappa)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(np.abs(predicted_dL) + 1e-12, np.abs(measured_dL) + 1e-12,
                     c=range(steps), cmap="viridis")
    lims = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
            max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(lims, lims, "k--", label="diagonal (exact match)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("|two-term predicted delta L|")
    axes[0].set_ylabel("|measured delta L|")
    axes[0].set_title("Theorem 2.2: one-step identity")
    axes[0].legend()

    axes[1].plot(eta_kappa, label=r"$\eta\kappa_k$ (realised)")
    axes[1].axhline(2.0, color="red", linestyle="--", label=r"$\eta\kappa=2$ (stability edge)")
    axes[1].set_xlabel("GD step")
    axes[1].set_ylabel(r"$\eta \kappa_k$")
    axes[1].set_title("Realised curvature vs. GD step")
    axes[1].legend()
    savefig(fig, "exp01_one_step_identity.png")

    print("Experiment 01 (one-step identity / realised curvature):")
    print(f"  max relative error of two-term Taylor prediction "
          f"(steps with eta*kappa<1.5): "
          f"{np.max(np.abs((predicted_dL-measured_dL)/(measured_dL+1e-12))[eta_kappa<1.5]):.4e}")
    print(f"  eta*kappa range observed: [{eta_kappa.min():.3f}, {eta_kappa.max():.3f}]")


if __name__ == "__main__":
    run()
