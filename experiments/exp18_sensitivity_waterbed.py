"""
Experiment 18 -- vizexp label: viz:waterbed (Section 8.3, Prop. 8.6)

Verifies Proposition 8.6 (Bode sensitivity integral / "waterbed"):
(1/2pi) * integral of log|S(e^{i omega})| over the unit circle is exactly
0 for S(z)=(z-1)/(z-p), |p|<1 -- reducing sensitivity at low frequency
necessarily raises it elsewhere.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    omegas = np.linspace(-np.pi, np.pi, 4000)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for eta_lambda in [0.5, 1.0, 1.5, 1.9]:
        p = 1 - eta_lambda
        z = np.exp(1j * omegas)
        S = (z - 1) / (z - p)
        logS = np.log(np.abs(S) + 1e-15)
        integral = np.trapezoid(logS, omegas) / (2 * np.pi)
        ax.plot(omegas, logS, label=f"eta*lambda={eta_lambda}  (integral={integral:.2e})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("omega"); ax.set_ylabel("log|S(e^{i omega})|")
    ax.set_title("Proposition 8.6: Bode sensitivity integral = 0 (waterbed effect)")
    ax.legend(fontsize=8)
    savefig(fig, "exp18_sensitivity_waterbed.png")

    print("Experiment 18 (Bode sensitivity waterbed):")
    for eta_lambda in [0.5, 1.0, 1.5, 1.9]:
        p = 1 - eta_lambda
        z = np.exp(1j * omegas)
        S = (z - 1) / (z - p)
        integral = np.trapezoid(np.log(np.abs(S) + 1e-15), omegas) / (2 * np.pi)
        print(f"  eta*lambda={eta_lambda}: (1/2pi)*integral log|S| = {integral:.3e} (predicted 0)")


if __name__ == "__main__":
    run()
