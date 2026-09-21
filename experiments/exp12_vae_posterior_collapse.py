"""
Experiment 12 -- vizexp label: viz:vae-collapse (Section 6.4.3, Prop. 6.13)

Verifies Proposition 6.13: the VAE Gaussian-KL regulariser has Hessian
diag(1, sigma^2/2) in (mu, logvar) coordinates -- curvature in the
log-variance channel vanishes proportionally to sigma^2, while the
*gradient* in that channel saturates at -1/2 rather than vanishing (so
"posterior collapse" is convergence to a strictly stable fixed point at
the prior, not a saturation of the restoring force).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def kl_term(mu, s):
    # KL(N(mu, e^s) || N(0,1)) = -0.5*(1 + s - mu^2 - e^s)
    return -0.5 * (1 + s - mu ** 2 - np.exp(s))


def run():
    mu0 = 0.3
    s_values = np.linspace(-10, 3, 200)
    dKL_ds = np.gradient(kl_term(mu0, s_values), s_values)
    d2KL_ds2 = np.gradient(dKL_ds, s_values)
    sigma2 = np.exp(s_values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(sigma2, dKL_ds, label="measured d(KL)/ds")
    axes[0].axhline(-0.5, color="r", ls="--", label="predicted limit = -1/2 (does not vanish)")
    axes[0].set_xlabel("sigma^2"); axes[0].set_xscale("log")
    axes[0].set_title("KL gradient saturates, does NOT vanish (Prop 6.13(2))")
    axes[0].legend()

    axes[1].plot(sigma2, d2KL_ds2, label="measured d^2(KL)/ds^2")
    axes[1].plot(sigma2, sigma2 / 2, "--", label="predicted sigma^2/2")
    axes[1].set_xlabel("sigma^2"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_title("KL curvature -> 0 as sigma^2 -> 0 (Proposition 6.10)")
    axes[1].legend()
    savefig(fig, "exp12_vae_posterior_collapse.png")

    print("Experiment 12 (VAE posterior collapse mechanism):")
    for s2 in [2.0, 1.0, 0.1, 0.01, 0.001]:
        s = np.log(s2)
        eps = 1e-4
        d1 = (kl_term(mu0, s + eps) - kl_term(mu0, s - eps)) / (2 * eps)
        d2 = (kl_term(mu0, s + eps) - 2 * kl_term(mu0, s) + kl_term(mu0, s - eps)) / eps ** 2
        print(f"  sigma^2={s2:<8} dKL/ds={d1:+.4f} (paper Script 33 predicts -> -0.5)   "
              f"d2KL/ds2={d2:.5f} (predicted {s2/2:.5f})")


if __name__ == "__main__":
    run()
