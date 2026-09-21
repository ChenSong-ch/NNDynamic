"""
Experiment 06 -- vizexp label: viz:alternating (Section 5, Prop. 5.3)

Dirac-GAN, simultaneous vs. alternating (Gauss-Seidel) GDA.
Verifies Proposition 5.3: det(A_alt) = 1 exactly for every eta, omega
(poles on the unit circle |z|=1, bounded orbit) vs. simultaneous updates
whose poles satisfy |z| = sqrt(1+eta^2*omega^2) > 1 (unbounded spiral).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig, plot_complex_poles


def run():
    omega, eta = 0.5, 0.3
    # simultaneous
    psi, theta = 0.5, 0.5
    traj_sim = []
    for _ in range(2000):
        traj_sim.append((psi, theta))
        dpsi = omega * theta
        dtheta = -omega * psi
        psi, theta = psi - eta * dpsi, theta - eta * dtheta
    traj_sim = np.array(traj_sim)

    # alternating (Gauss-Seidel): update psi with current theta, then theta with NEW psi
    psi, theta = 0.5, 0.5
    traj_alt = []
    for _ in range(2000):
        traj_alt.append((psi, theta))
        psi_new = psi - eta * omega * theta
        theta_new = theta - eta * (-omega * psi_new)
        psi, theta = psi_new, theta_new
    traj_alt = np.array(traj_alt)

    A_sim = np.array([[1, eta * omega], [-eta * omega, 1]])
    A_alt = np.array([[1, eta * omega], [-eta * omega, 1 - eta ** 2 * omega ** 2]])
    z_sim = np.linalg.eigvals(A_sim)
    z_alt = np.linalg.eigvals(A_alt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(traj_sim[:, 0], traj_sim[:, 1], lw=0.7, label="simultaneous")
    axes[0].plot(traj_alt[:, 0], traj_alt[:, 1], lw=0.7, label="alternating")
    axes[0].set_xlabel("psi"); axes[0].set_ylabel("theta")
    axes[0].set_title("Dirac-GAN trajectories"); axes[0].legend()

    plot_complex_poles(axes[1], z_sim, label="simultaneous poles", color="C0")
    plot_complex_poles(axes[1], z_alt, label="alternating poles", color="C1", marker="^")
    axes[1].set_title("Poles: alternating stays exactly on |z|=1")
    axes[1].legend(fontsize=8)

    r_sim = np.linalg.norm(traj_sim, axis=1)
    r_alt = np.linalg.norm(traj_alt, axis=1)
    axes[2].semilogy(r_sim, label="||(psi,theta)|| simultaneous")
    axes[2].semilogy(r_alt, label="||(psi,theta)|| alternating")
    axes[2].set_xlabel("step"); axes[2].set_title("radius over training")
    axes[2].legend()
    savefig(fig, "exp06_alternating_vs_simultaneous.png")

    print("Experiment 06 (alternating vs simultaneous GDA):")
    print(f"  det(A_sim) = {np.linalg.det(A_sim):.6f}, |z_sim| = {np.abs(z_sim)}")
    print(f"  det(A_alt) = {np.linalg.det(A_alt):.6f} (predicted exactly 1), |z_alt| = {np.abs(z_alt)}")
    print(f"  final radius: simultaneous={r_sim[-1]:.3e}  alternating={r_alt[-1]:.3e}")


if __name__ == "__main__":
    run()
