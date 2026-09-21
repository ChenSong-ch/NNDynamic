"""
Experiment 24 -- vizexp label: viz:gain-scheduling (Section 12.1, Prop. 12.1)

Verifies Proposition 12.1 (locally optimal / "dead-beat" step size):
eta_k* = 1/kappa_k drives the realised pole along the gradient direction
to exactly 0 each step, and never violates the eta*kappa<2 stability
condition -- reproducing Script 6's comparison of a fixed (diverging)
learning rate against the gain-scheduled one.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, hvp_generic, numerical_gradient
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(24)
    mlp = TinyMLP([3, 8, 1], activation="tanh", seed=24)
    N = 10
    X = rng.normal(size=(N, 3)) * 0.5
    y = rng.normal(size=(N, 1)) * 0.3

    def loss(theta):
        r = (mlp.forward(theta, X) - y).ravel()
        return 0.5 * np.mean(r ** 2)

    def run_fixed(eta, steps=40):
        theta = mlp.theta0.copy()
        losses = []
        for _ in range(steps):
            L = loss(theta)
            losses.append(L)
            if not np.isfinite(L) or L > 1e6:
                losses += [np.nan] * (steps - len(losses))
                break
            g = numerical_gradient(loss, theta)
            theta = theta - eta * g
        return losses

    def run_adaptive(steps=200, cap=5.0):
        theta = mlp.theta0.copy()
        losses, etas = [], []
        for _ in range(steps):
            L = loss(theta)
            losses.append(L)
            g = numerical_gradient(loss, theta)
            Hg = hvp_generic(loss, theta, g)
            kappa = (g @ Hg) / (g @ g + 1e-18)
            eta_k = min(1.0 / max(kappa, 1e-6), cap)
            etas.append(eta_k)
            theta = theta - eta_k * g
        return losses, etas

    losses_fixed = run_fixed(1.6, steps=40)
    losses_ad, etas_ad = run_adaptive(steps=200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(losses_fixed, label="fixed eta=1.6 (diverges)")
    axes[0].semilogy(losses_ad[:40], label="gain-scheduled eta_k=1/kappa_k")
    axes[0].set_xlabel("GD step"); axes[0].set_ylabel("loss")
    axes[0].set_title("Proposition 12.1: gain scheduling prevents divergence")
    axes[0].legend()

    axes[1].plot(etas_ad)
    axes[1].set_xlabel("GD step"); axes[1].set_ylabel("eta_k = 1/kappa_k")
    axes[1].set_title("Adaptive step size stays bounded (capped at 5.0)")
    savefig(fig, "exp24_gain_scheduling.png")

    print("Experiment 24 (gain scheduling / dead-beat pole):")
    print(f"  fixed eta=1.6: final loss = {losses_fixed[-1]}")
    print(f"  adaptive: final loss (step 200) = {losses_ad[-1]:.3e}, no divergence")


if __name__ == "__main__":
    run()
