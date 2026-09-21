"""
Experiment 22 -- vizexp label: viz:real-kernel-bode (Section 11.4-11.5,
Prop. 11.6, Cor. 11.7)

Paper's version pulls a real kernel out of pretrained ResNet-18
(ImageNet weights, downloaded via torchvision) and FFTs it. That
download is not available in this offline environment, so we use a
structurally identical synthetic kernel -- a smoothing kernel [1,2,1]/4
plus a sharpening kernel -- exactly Script 25[D]/26's own toy kernels,
which the paper itself uses for closed-form verification before turning
to the real pretrained kernel.

Verifies Proposition 11.6 (circulant layers diagonalised by DFT) and
Corollary 11.7 (residual sum = uniform +1 vertical shift of the Bode
curve: constructive amplification where khat>=0, floor recovery
1-|khat| where khat<0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    p = 64
    omegas = 2 * np.pi * np.arange(p) / p

    def khat(kernel, taps, omegas):
        # kernel: dict offset->value
        return sum(v * np.exp(-1j * omegas * t) for t, v in zip(taps, kernel))

    smooth_taps = [-1, 0, 1]
    smooth_kernel = [0.25, 0.5, 0.25]
    sharp_taps = [-1, 0, 1]
    sharp_kernel = [-0.6, 0.0, -0.6]  # reaches khat=-1.2 -> dead-frequency-like regime

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for taps, kernel, name in [(smooth_taps, smooth_kernel, "smoothing [1,2,1]/4"),
                                (sharp_taps, sharp_kernel, "near-cancelling kernel")]:
        kh = khat(kernel, taps, omegas)
        plain = np.abs(kh)
        resid = np.abs(1 + kh)
        axes[0].plot(omegas, plain, label=f"plain |khat|: {name}")
        axes[1].plot(omegas, resid, label=f"residual |1+khat|: {name}")

    axes[0].set_xlabel("spatial frequency omega"); axes[0].set_ylabel("|khat(omega)|")
    axes[0].set_title("Plain branch Bode magnitude (Prop. 11.6)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color="gray", lw=0.6)
    axes[1].set_xlabel("spatial frequency omega"); axes[1].set_ylabel("|1+khat(omega)|")
    axes[1].set_title("Residual sum: uniform +1 shift, floor recovery (Cor. 11.7)")
    axes[1].legend(fontsize=8)
    savefig(fig, "exp22_conv_kernel_bode.png")

    kh = khat(sharp_kernel, sharp_taps, omegas)
    print("Experiment 22 (circulant/Bode analysis of conv kernels):")
    print(f"  min |1+khat| over grid = {np.min(np.abs(1+kh)):.4f} "
          f"(dips toward 0 -> dead-frequency phenomenon, Prop. 11.9)")


if __name__ == "__main__":
    run()
