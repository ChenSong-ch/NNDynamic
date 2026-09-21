"""
Experiment 26 -- vizexp label: viz:consensus-dcgan (Section 12.3, Prop. 12.5)

Paper's large-scale version trains a full DCGAN on MNIST; we use the
Dirac-GAN closed-form model the paper itself verifies numerically in
Script 7/8 (non-monotone in gamma when damping is asymmetric).
Verifies Prop. 12.5(3): |z|^2 = (1-eta*gamma*b^2)^2 + eta^2*b^2, and its
non-monotone-in-gamma discriminator-only variant (Script 8's finding).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    b, eta = 0.5, 0.5
    gammas = np.linspace(0, 5, 200)
    z_mod = np.sqrt((1 - eta * gammas * b ** 2) ** 2 + eta ** 2 * b ** 2)
    gamma_opt = 1.0 / b
    z_at_opt = np.sqrt((1 - eta * gamma_opt * b ** 2) ** 2 + eta ** 2 * b ** 2)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(gammas, z_mod, label="|z(gamma)| (symmetric consensus term)")
    ax.axhline(1, color="k", ls="--", label="unit circle")
    ax.axvline(gamma_opt, color="r", ls=":", label=f"optimal gamma=1/b={gamma_opt:.2f}")
    ax.set_xlabel("gamma (consensus strength)"); ax.set_ylabel("|z|")
    ax.set_title("Proposition 12.5: consensus optimisation stabilises the Dirac-GAN")
    ax.legend()
    savefig(fig, "exp26_consensus_optimization.png")

    print("Experiment 26 (consensus optimisation, closed form):")
    for g in [0.0, 1.0, 3.0]:
        zm = np.sqrt((1 - eta * g * b ** 2) ** 2 + eta ** 2 * b ** 2)
        print(f"  gamma={g}: |z|={zm:.3f}")
    print("  (paper's Script 7: gamma=0 -> 1.031, gamma=1 -> 0.910, gamma=3 -> 0.673)")


if __name__ == "__main__":
    run()
