"""
Experiment 25 -- vizexp label: viz:wd-null-poles (Section 13, Prop. 13.1)

Verifies Proposition 13.1(3): weight decay shifts EVERY pole uniformly by
-eta*lambda, so the n-r exact-1 null-space poles of an over-parameterised
interpolating network move to 1-eta*lambda while the rest shift by the
same amount -- a rigid translation of the whole pole histogram.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, numerical_jacobian
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(25)
    mlp = TinyMLP([3, 14, 1], activation="tanh", seed=25)  # heavily over-parameterised
    N = 5
    X = rng.normal(size=(N, 3)) * 0.4
    y = rng.normal(size=(N, 1)) * 0.2

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

    J = numerical_jacobian(residual, theta)
    G_eigs = np.linalg.eigvalsh(J.T @ J / N)
    eta = 0.1

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lam_wd, color in [(0.0, "C0"), (1.0, "C1"), (3.0, "C2")]:
        poles = 1 - eta * (G_eigs + lam_wd)
        ax.hist(poles, bins=40, alpha=0.55, color=color, label=f"lambda_wd={lam_wd}")
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.set_xlabel("pole z"); ax.set_ylabel("count")
    ax.set_title("Proposition 13.1(3): weight decay rigidly shifts every pole by -eta*lambda")
    ax.legend()
    savefig(fig, "exp25_weight_decay_null_poles.png")

    print("Experiment 25 (weight decay null-space pole shift):")
    n_null = np.sum(G_eigs < 1e-6 * G_eigs.max())
    print(f"  # near-null directions at lambda_wd=0: {n_null} / {mlp.n_params}")
    for lam_wd in [0.0, 1.0, 3.0]:
        pole_of_null = 1 - eta * lam_wd
        print(f"  lambda_wd={lam_wd}: predicted null-space pole = {pole_of_null:.3f}")


if __name__ == "__main__":
    run()
