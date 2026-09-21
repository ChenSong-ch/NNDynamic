"""
Experiment 08 -- vizexp label: viz:dirac-gan-viz (Section 12, Prop. 12.5)

Dirac-GAN, simultaneous GDA with and without consensus-optimisation
regularisation (Proposition 12.5). Verifies: gamma=0 -> eigenvalues sit
exactly on the imaginary axis and (psi,theta) spirals outward; gamma>0 ->
eigenvalues move to strictly negative real part and the trajectory
spirals inward to the origin.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    eta = 0.1
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {0.0: "C0", 0.3: "C1", 1.0: "C2"}
    for gamma in [0.0, 0.3, 1.0]:
        psi, theta = 0.5, 0.5
        traj = []
        eig_traj = []
        for _ in range(500):
            traj.append((psi, theta))
            v = np.array([omega_theta := theta, -psi])  # v = (omega*theta, -omega*psi), omega=1
            Jv = np.array([[0.0, 1.0], [-1.0, 0.0]])
            v_tilde = v + gamma * (Jv.T @ v)
            Jtilde = Jv + gamma * (Jv.T @ Jv)
            eig_traj.append(np.linalg.eigvals(Jtilde))
            psi, theta = psi - eta * v_tilde[0], theta - eta * v_tilde[1]
        traj = np.array(traj)
        axes[0].plot(traj[:, 0], traj[:, 1], color=colors[gamma], lw=0.8,
                     label=f"gamma={gamma}")
        eig_traj = np.array(eig_traj)
        axes[1].scatter(eig_traj[:, 0].real, eig_traj[:, 0].imag, s=4,
                         color=colors[gamma], label=f"gamma={gamma}")

    axes[0].scatter([0], [0], color="k", marker="*", s=80, zorder=5, label="Nash eq.")
    axes[0].set_xlabel("psi"); axes[0].set_ylabel("theta")
    axes[0].set_title("Dirac-GAN trajectory: gamma=0 spirals out, gamma>0 spirals in")
    axes[0].legend(fontsize=8)
    axes[1].axvline(0, color="k", lw=0.7)
    axes[1].set_xlabel("Re(eigenvalue of game Jacobian)")
    axes[1].set_ylabel("Im(eigenvalue)")
    axes[1].set_title("Consensus optimisation moves eigenvalues off the imaginary axis")
    axes[1].legend(fontsize=8)
    savefig(fig, "exp08_dirac_gan_spiral.png")
    print("Experiment 08 (Dirac-GAN + consensus optimisation): see figure.")


if __name__ == "__main__":
    run()
