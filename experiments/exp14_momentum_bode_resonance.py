"""
Experiment 14 -- vizexp label: viz:momentum-bode (Section 8.4, Prop. 8.5)

Reproduces Script 10 of the paper exactly: closed-form Bode resonance
frequency of heavy-ball momentum's noise-transfer function, vs. plain
GD's monotone (non-resonant) noise transfer.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    eta, beta, lam = 0.05, 0.9, 8.0
    p = 1 + beta - eta * lam

    def T_GD_mag(w):
        z = np.exp(1j * w)
        return np.abs(-eta / (z - 1 + eta * lam))

    def T_HB_mag(w):
        z = np.exp(1j * w)
        return np.abs(-eta * z / (z ** 2 - p * z + beta))

    omegas = np.linspace(1e-3, np.pi, 2000)
    gd_mag = T_GD_mag(omegas)
    hb_mag = T_HB_mag(omegas)

    cos_wr = p * (1 + beta) / (4 * beta)
    omega_r = np.arccos(np.clip(cos_wr, -1, 1))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(omegas, gd_mag, label="|T_GD(e^{i omega})| (monotone, no peak)")
    ax.plot(omegas, hb_mag, label="|T_HB(e^{i omega})| (heavy-ball)")
    ax.axvline(omega_r, color="r", ls="--", label=f"predicted resonance omega_r={omega_r:.4f}")
    ax.set_xlabel("omega"); ax.set_ylabel("|T(e^{i omega})|")
    ax.set_title("Proposition 8.5: closed-form Bode resonance of heavy-ball momentum")
    ax.legend()
    savefig(fig, "exp14_momentum_bode_resonance.png")

    idx = np.argmax(hb_mag)
    ratio = hb_mag[idx] / T_GD_mag(omegas[idx])
    print("Experiment 14 (momentum Bode resonance):")
    print(f"  predicted omega_r = {omega_r:.4f}, grid-argmax omega = {omegas[idx]:.4f}")
    print(f"  gain at omega_r: HB={T_HB_mag(np.array([omega_r]))[0]:.4f}  "
          f"GD={T_GD_mag(np.array([omega_r]))[0]:.4f}  ratio={T_HB_mag(np.array([omega_r]))[0]/T_GD_mag(np.array([omega_r]))[0]:.2f}x")
    print("  (paper's Script 10: omega_r=0.6573, ratio=10.46x)")


if __name__ == "__main__":
    run()
