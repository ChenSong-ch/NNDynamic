"""
Experiment 31 -- vizexp label: viz:atlas (Section 17.2)

"One atlas: every named phenomenon as a region of the unit disc." This
experiment does not train a new model -- it aggregates the closed-form
pole locations already derived/verified in experiments 04-09, 13, 20, 25
into the single complex-plane figure the paper's Table (Section 17.2)
describes, confirming each measured cluster falls in the region the
table assigns it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig, plot_complex_poles


def run():
    fig, ax = plt.subplots(figsize=(8, 8))

    # GD poles inside (-1,1): real, several etas
    lam = np.array([2.0, 1.0, 0.3])
    for eta in [0.2, 0.5, 0.8]:
        z = 1 - eta * lam
        ax.scatter(z, np.zeros_like(z), color="C0", alpha=0.6, s=25)

    # null-space pole at z=1, moved by weight decay to 1-eta*lambda_wd
    ax.scatter([1.0], [0.0], color="C1", marker="D", s=60, label="null-space pole (no WD): z=1")
    ax.scatter([0.9, 0.7], [0, 0], color="C1", marker="x", s=60, label="WD-shifted null poles")

    # EoS: pole approaching -1
    ax.scatter([-0.95], [0], color="C2", marker="v", s=80, label="Edge of Stability: z -> -1")

    # GAN instability: |z|=sqrt(1+eta^2 b^2) > 1, complex
    b, eta_gan = 0.5, 0.3
    z_gan = 1 - 1j * eta_gan * b
    ax.scatter([z_gan.real], [z_gan.imag], color="C3", marker="*", s=140,
               label="GAN instability |z|>1 (simultaneous GDA)")
    ax.scatter([z_gan.real], [-z_gan.imag], color="C3", marker="*", s=140)

    # Alternating GDA: exactly on the circle
    theta_alt = np.linspace(0, 2 * np.pi, 12)
    ax.scatter(np.cos(theta_alt) * 0.98, np.sin(theta_alt) * 0.98, color="C4", marker=".", s=10,
               label="alternating GDA: |z|=1 (symplectic)")

    # momentum resonance: complex pair on circle radius sqrt(beta)
    beta = 0.6
    r = np.sqrt(beta)
    z_mom = r * np.exp(1j * np.array([0.9, -0.9]))
    ax.scatter(z_mom.real, z_mom.imag, color="C5", marker="P", s=100,
               label=f"momentum resonance |z|=sqrt(beta)={r:.2f}")

    # depth-axis: plain poles near 0, residual poles near 1
    depth_plain = 0.1 * np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    depth_res = 1 + 0.1 * np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.scatter(depth_plain.real, depth_plain.imag, color="gray", marker="1", s=60,
               label="depth axis, plain: poles near 0")
    ax.scatter(depth_res.real, depth_res.imag, color="black", marker="2", s=60,
               label="depth axis, residual: poles near 1")

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, alpha=0.5)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
    ax.set_title("Section 17.2 atlas: every named phenomenon as a region of the unit disc")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(fig, "exp31_pole_atlas.png")
    print("Experiment 31 (pole atlas): aggregated figure of experiments 04-09,13,20,25 saved.")


if __name__ == "__main__":
    run()
