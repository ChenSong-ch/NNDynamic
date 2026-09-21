"""
Experiment 13 -- vizexp label: viz:momentum-poles (Section 4.4, Prop. 4.7)

Heavy-ball momentum poles: z_{1,2} roots of
  chi_HB(z) = z^2 - (1+beta-eta*lambda) z + beta.
Verifies Prop. 4.7: real & distinct for small beta, transitioning at the
critical beta* (where discriminant=0) to a complex-conjugate pair sitting
exactly on the circle of radius sqrt(beta).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig, plot_complex_poles


def run():
    eta, lam = 0.05, 8.0
    betas = np.linspace(0.001, 0.99, 60)
    all_poles = []
    for beta in betas:
        p = 1 + beta - eta * lam
        roots = np.roots([1, -p, beta])
        all_poles.append(roots)
    all_poles = np.array(all_poles)

    # critical beta*: (1+beta-eta*lam)^2 = 4*beta
    from numpy.polynomial import polynomial as P
    # solve (1+b-eta*lam)^2-4b=0 for b via numpy roots on the poly in b
    # (1-eta*lam+b)^2 - 4b = b^2 + (2*(1-eta*lam)-4) b + (1-eta*lam)^2
    c = 1 - eta * lam
    coeffs = [1, 2 * c - 4, c ** 2]
    beta_star_candidates = np.roots(coeffs)
    beta_star = [b.real for b in beta_star_candidates if 0 < b.real < 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc = axes[0].scatter(all_poles.real, all_poles.imag, c=np.repeat(betas, 2), cmap="viridis", s=8)
    theta = np.linspace(0, 2 * np.pi, 200)
    if beta_star:
        r = np.sqrt(beta_star[0])
        axes[0].plot(r * np.cos(theta), r * np.sin(theta), "r--", label=f"radius sqrt(beta*)={r:.3f}")
    axes[0].plot(np.cos(theta), np.sin(theta), "k:", lw=0.7)
    axes[0].set_xlabel("Re(z)"); axes[0].set_ylabel("Im(z)")
    axes[0].set_title("Heavy-ball poles vs. beta (color = beta)")
    axes[0].legend()
    plt.colorbar(sc, ax=axes[0], label="beta")

    axes[1].plot(betas, np.abs(all_poles[:, 0]), label="|z1|")
    axes[1].plot(betas, np.abs(all_poles[:, 1]), label="|z2|")
    axes[1].plot(betas, np.sqrt(betas), "--", label="sqrt(beta) (predicted, complex regime)")
    if beta_star:
        axes[1].axvline(beta_star[0], color="r", ls=":", label=f"beta*={beta_star[0]:.3f}")
    axes[1].set_xlabel("beta"); axes[1].set_ylabel("|pole|")
    axes[1].set_title("Modulus transition at beta* (Proposition 4.7)")
    axes[1].legend()
    savefig(fig, "exp13_momentum_poles.png")

    print("Experiment 13 (momentum poles, real-to-complex transition):")
    print(f"  critical beta* = {beta_star}")


if __name__ == "__main__":
    run()
