"""
Experiment 37 -- vizexp label: viz:grok (Appendix A.1, Prop. A.1 / Script 31)

Verifies the exact scaling law of Proposition A.1: the null-space
component under weight decay decays as c_k=(1-eta*lambda)^k*c0, giving
time-to-threshold t_eps = ln(1/eps)/(eta*lambda) ~ ln(1/eps)/lambda for
small eta*lambda, i.e. t_eps * lambda = constant. This is the paper's
own proposed *mechanism* behind grokking's weight-decay dependence
(reproducing Script 31[A] exactly); testing it against a real grokking
benchmark (modular-arithmetic Transformer) is listed as future work E1
in the paper's own appendix and is out of scope for a laptop-only repo.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    eta, eps = 0.1, 0.01
    lambdas = np.array([0.02, 0.05, 0.1, 0.2])
    c0 = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t_eps_measured = []
    for lam in lambdas:
        c = c0
        k = 0
        while abs(c) > eps * c0 and k < 200000:
            c *= (1 - eta * lam)
            k += 1
        t_eps_measured.append(k)
        ks = np.arange(0, k + 20)
        axes[0].semilogy(ks, np.abs(c0 * (1 - eta * lam) ** ks), label=f"lambda={lam}")

    t_eps_measured = np.array(t_eps_measured)
    t_eps_pred = np.log(1 / eps) / (-np.log(1 - eta * lambdas))

    axes[0].axhline(eps, color="k", ls="--", label="threshold eps")
    axes[0].set_xlabel("GD step k"); axes[0].set_ylabel("null-space component |c_k|")
    axes[0].set_title("Proposition A.1: null-space contraction under weight decay")
    axes[0].legend(fontsize=7)

    axes[1].plot(lambdas, t_eps_measured, "o-", label="measured t_eps")
    axes[1].plot(lambdas, t_eps_pred, "x--", label="predicted ln(1/eps)/(-ln(1-eta*lambda))")
    axes[1].set_xlabel("lambda (weight decay)"); axes[1].set_ylabel("t_eps (steps to threshold)")
    axes[1].set_title("t_eps * lambda ~ constant (grokking-time scaling law)")
    axes[1].legend()
    savefig(fig, "exp37_grokking_time_law.png")

    print("Experiment 37 (grokking time-to-threshold law):")
    for lam, t_m, t_p in zip(lambdas, t_eps_measured, t_eps_pred):
        print(f"  lambda={lam}: measured t_eps={t_m}  predicted={t_p:.1f}  t_eps*lambda={t_m*lam:.2f}")
    print("  (paper's Script 31[A]: t_eps*lambda constant at 46.05 across a 10x sweep)")


if __name__ == "__main__":
    run()
