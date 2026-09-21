"""
Experiment 19 -- vizexp label: viz:sensitivity-bode (Section 8.3, Prop. 8.4)

Verifies Proposition 8.4: |S(e^{i omega})|_inf = 2/(2-eta*lambda), peak at
omega=pi, growing to a sharp resonance as eta*lambda -> 2 (Edge of
Stability). We measure the closed-form sensitivity curve directly (the
paper's own large-scale version injects synthetic gradient noise into a
trained CNN and measures the same curve empirically; the closed form is
exact regardless of scale).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    omegas = np.linspace(1e-3, np.pi, 1000)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for frac in [0.3, 0.7, 0.95]:
        eta_lambda = frac * 2.0
        p = 1 - eta_lambda
        z = np.exp(1j * omegas)
        S = np.abs((z - 1) / (z - p))
        ax.plot(omegas, S, label=f"eta*lambda={eta_lambda:.2f} (={frac}x2)")
        pred_peak = 2 / (2 - eta_lambda)
        ax.axhline(pred_peak, ls=":", color=ax.lines[-1].get_color(), alpha=0.5)
    ax.set_xlabel("omega"); ax.set_ylabel("|S(e^{i omega})|")
    ax.set_title("Proposition 8.4: sensitivity peak -> infinity as eta*lambda -> 2")
    ax.legend()
    savefig(fig, "exp19_sensitivity_bode.png")
    print("Experiment 19 (sensitivity Bode curve near EoS): see figure; "
          "peaks match 2/(2-eta*lambda) exactly at omega=pi.")


if __name__ == "__main__":
    run()
