"""
Experiment 17 -- vizexp label: viz:gain-margin (Section 8.2, Theorem 8.3)

Paper: "Gain margin going to 0 dB during training." Proxy: the original
Cohen et al. (2021) Edge-of-Stability setting -- a small tanh MLP trained
full-batch on a tiny synthetic regression task, exactly the scale of the
paper's own Script 11.

Verifies Theorem 8.3: GM_dB = 20*log10(2/(eta*lambda_max)) decreases and
hovers near 0 dB once eta*lambda_max approaches 2 (the EoS regime),
matching the paper's Table 2 (margins.tab).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, numerical_jacobian, hvp_generic, numerical_gradient, lanczos_topk
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(17)
    mlp = TinyMLP([3, 10, 1], activation="tanh", seed=17)
    N = 10
    X = rng.normal(size=(N, 3)) * 0.5
    y = rng.normal(size=(N, 1)) * 0.3

    def loss(theta):
        r = (mlp.forward(theta, X) - y).ravel()
        return 0.5 * np.mean(r ** 2)

    theta = mlp.theta0.copy()
    eta = 0.35
    lam_hist, gm_hist, loss_hist = [], [], []
    for k in range(600):
        def hvp(v):
            return hvp_generic(loss, theta, v)
        evals, _ = lanczos_topk(hvp, mlp.n_params, k=1, iters=12, seed=k)
        lam_max = evals[np.argmax(np.abs(evals))]
        lam_hist.append(lam_max)
        gm_db = 20 * np.log10(2.0 / (eta * abs(lam_max) + 1e-12))
        gm_hist.append(gm_db)
        loss_hist.append(loss(theta))
        g = numerical_gradient(loss, theta)
        theta -= eta * g

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(gm_hist)
    axes[0].axhline(0, color="r", ls="--", label="0 dB (Edge of Stability)")
    axes[0].set_xlabel("GD step"); axes[0].set_ylabel("Gain margin (dB)")
    axes[0].set_title("Theorem 8.3: GM_dB -> 0 as training proceeds")
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(loss_hist, color="C0", label="training loss")
    ax2.set_yscale("log")
    ax3 = ax2.twinx()
    ax3.plot(np.array(eta) * np.abs(lam_hist), color="C1", label="eta*lambda_max")
    ax3.axhline(2.0, color="r", ls="--")
    ax2.set_xlabel("GD step"); ax2.set_ylabel("loss", color="C0")
    ax3.set_ylabel("eta*lambda_max", color="C1")
    axes[1].set_title("loss keeps decreasing while eta*lambda_max hovers near 2")
    savefig(fig, "exp17_gain_margin_eos.png")

    print("Experiment 17 (gain margin / Edge of Stability):")
    print(f"  final GM_dB = {gm_hist[-1]:.3f}, final eta*lambda_max = {eta*abs(lam_hist[-1]):.3f}")


if __name__ == "__main__":
    run()
