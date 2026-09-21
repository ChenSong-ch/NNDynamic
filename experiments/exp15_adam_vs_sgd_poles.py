"""
Experiment 15 -- vizexp label: viz:adam-poles (Section 9, Prop. 9.4)

Verifies Proposition 9.4: for diagonal Hessian H=diag(h_i) and diagonal
preconditioner D=diag(h_i^{-1/2}), the realised curvature along the
Adam-like direction Dg/||Dg|| never exceeds that along the raw gradient
g/||g|| (Cauchy-Schwarz), so Adam's poles sit closer to 0 than SGD's for
the same nominal eta.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(15)
    n = 30
    h = np.abs(rng.normal(1.0, 0.8, size=n)) + 0.05  # diagonal Hessian eigenvalues
    trials = 200
    kappa_gd, kappa_adam = [], []
    for _ in range(trials):
        g = rng.normal(size=n)
        u_gd = g / np.linalg.norm(g)
        D = 1.0 / np.sqrt(h)
        u_adam = (D * g) / np.linalg.norm(D * g)
        kappa_gd.append(u_gd @ (h * u_gd))
        kappa_adam.append(u_adam @ (h * u_adam))
    kappa_gd = np.array(kappa_gd)
    kappa_adam = np.array(kappa_adam)

    eta = 0.15
    z_gd = 1 - eta * kappa_gd
    z_adam = 1 - eta * kappa_adam

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(kappa_gd, kappa_adam, s=10, alpha=0.6)
    lims = [0, max(kappa_gd.max(), kappa_adam.max())]
    axes[0].plot(lims, lims, "k--", label="kappa_Adam = kappa_GD")
    axes[0].set_xlabel("kappa_GD (realised curvature, raw gradient)")
    axes[0].set_ylabel("kappa_Adam (diagonally preconditioned)")
    axes[0].set_title("Proposition 9.4: kappa_Adam <= kappa_GD always")
    axes[0].legend()

    axes[1].hist(z_gd, bins=25, alpha=0.6, label="SGD poles 1-eta*kappa_GD")
    axes[1].hist(z_adam, bins=25, alpha=0.6, label="Adam poles 1-eta*kappa_Adam")
    axes[1].set_xlabel("realised pole z"); axes[1].set_title("Adam poles compressed toward the centre")
    axes[1].legend()
    savefig(fig, "exp15_adam_vs_sgd_poles.png")

    print("Experiment 15 (Adam vs SGD realised curvature/poles):")
    print(f"  violations of kappa_Adam <= kappa_GD out of {trials}: "
          f"{np.sum(kappa_adam > kappa_gd + 1e-9)}")
    print(f"  mean kappa_GD={kappa_gd.mean():.4f}  mean kappa_Adam={kappa_adam.mean():.4f}")


if __name__ == "__main__":
    run()
