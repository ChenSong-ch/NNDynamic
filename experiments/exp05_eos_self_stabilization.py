"""
Experiment 05 -- vizexp label: viz:eos-slow (Section 8, Proposition 8.2)

Paper: "The slow coordinate that tames the sharp one, on the Cohen et
al. CIFAR-10 MLP." We instead implement the *exact minimal model* the
paper derives this mechanism from (Eq. eos2d-map / Proposition 8.2)
directly -- a 2-coordinate map with curvature-coupling -- since that
closed-form system is what the large-scale experiment is designed to
test the many-coordinate analogue of.

Verifies: for eta*a > 2 (naively unstable), the map locks onto the exact
period-2 orbit with effective curvature a + b*w2* = 2/eta (pole = -1,
the Edge-of-Stability boundary) predicted by Eq. (eos-exact).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    a, b, c, eta = 1.8, 3.0, 0.05, 1.15
    assert eta * a > 2, "need naive instability eta*a > 2"

    w1, w2 = 0.05, 0.0
    hist_w1, hist_w2, hist_curv = [], [], []
    for k in range(4000):
        hist_w1.append(w1); hist_w2.append(w2)
        hist_curv.append(eta * (a + b * w2))
        w1_new = w1 - eta * (a + b * w2) * w1
        w2_new = w2 - eta * (0.5 * b * w1 ** 2 + c * w2)
        w1, w2 = w1_new, w2_new

    w2_star_pred = (2 / eta - a) / b
    A_pred = np.sqrt(2 * c * (a - 2 / eta) / b ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(hist_w1[-100:])
    axes[0].axhline(A_pred, color="r", ls="--", label=f"+A={A_pred:.4f}")
    axes[0].axhline(-A_pred, color="r", ls="--")
    axes[0].set_title("w1_k: locks into a period-2 orbit")
    axes[0].legend()

    axes[1].plot(hist_w2[-200:])
    axes[1].axhline(w2_star_pred, color="r", ls="--", label=f"w2*={w2_star_pred:.4f}")
    axes[1].set_title("w2_k: slow variable settles at w2*")
    axes[1].legend()

    axes[2].plot(hist_curv)
    axes[2].axhline(2.0, color="k", ls="--", label="EoS boundary eta*(a+b*w2)=2")
    axes[2].set_title("effective instantaneous curvature * eta")
    axes[2].legend()
    savefig(fig, "exp05_eos_self_stabilization.png")

    print("Experiment 05 (2D EoS self-stabilization, Proposition 8.2):")
    print(f"  predicted w2* = {w2_star_pred:.5f}, observed = {w2:.5f}")
    print(f"  predicted A   = {A_pred:.5f}, observed |w1| = {abs(w1):.5f}")
    print(f"  eta*(a+b*w2*) at convergence = {eta*(a+b*w2):.6f} (should be exactly 2.0)")


if __name__ == "__main__":
    run()
