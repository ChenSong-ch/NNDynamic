"""
Experiment 39 -- vizexp label: viz:forgetting (Appendix A.3, Prop. A.3)

Reproduces Scripts 31[B] and 38: sequential fine-tuning drift
delta_ss = -H_2^{-1} g_2 causes a quadratic increase in task-1 loss;
adding an EWC penalty lambda_ewc*F_1 monotonically decreases the "drift
energy" E(lambda_ewc)=g_2^T(H_2+lambda_ewc*F_1)^{-1}g_2 (Prop. A.3(1)),
even though the raw drift NORM ||delta_ss|| need not be monotone when
H_2 and F_1 don't commute (Script 38's explicit counterexample).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(39)
    n = 5
    A1 = rng.normal(size=(n, n)); H1 = A1.T @ A1 / n + 0.1 * np.eye(n)
    A2 = rng.normal(size=(n, n)); F1 = A1.T @ A1 / n + 0.05 * np.eye(n)  # Fisher approx of task 1
    H2 = A2.T @ A2 / n + 0.1 * np.eye(n)
    g2 = rng.normal(size=n)

    lambdas = np.linspace(0, 10, 60)
    drift_norms, drift_energy, task1_loss_incr = [], [], []
    for lam in lambdas:
        M = H2 + lam * F1
        delta = -np.linalg.solve(M, g2)
        drift_norms.append(np.linalg.norm(delta))
        drift_energy.append(g2 @ np.linalg.solve(M, g2))
        task1_loss_incr.append(0.5 * delta @ H1 @ delta)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(lambdas, drift_energy, label="drift energy E(lambda_ewc) [monotone, Prop A.3(1)]")
    axes[0].plot(lambdas, task1_loss_incr, label="task-1 loss increase (Eq. forget-loss)")
    axes[0].set_xlabel("lambda_ewc"); axes[0].set_yscale("log")
    axes[0].set_title("EWC suppresses forgetting (both quantities decreasing)")
    axes[0].legend(fontsize=8)

    axes[1].plot(lambdas, drift_norms, color="C2")
    axes[1].set_xlabel("lambda_ewc"); axes[1].set_ylabel("||delta_ss^ewc||")
    axes[1].set_title("Raw drift NORM need not be monotone (non-commuting H2,F1)")
    savefig(fig, "exp39_catastrophic_forgetting_ewc.png")

    no_ewc = task1_loss_incr[0]
    with_ewc = task1_loss_incr[np.argmin(np.abs(lambdas - 5.0))]
    print("Experiment 39 (catastrophic forgetting / EWC):")
    print(f"  ||delta_ss|| no EWC = {drift_norms[0]:.4f}, at lambda_ewc=5: {drift_norms[np.argmin(np.abs(lambdas-5.0))]:.4f}")
    print(f"  task-1 loss increase: no EWC = {no_ewc:.4f}, at lambda_ewc=5: {with_ewc:.4f} "
          f"({no_ewc/with_ewc:.1f}x reduction)")
    non_monotone = np.any(np.diff(drift_norms) > 1e-9) and np.any(np.diff(drift_norms) < -1e-9)
    print(f"  drift norm non-monotone somewhere in the sweep: {non_monotone} "
          f"(paper's Script 38 exhibits exactly this)")
    print("  (paper's Script 31[B]: 5.4x drift reduction, 28.1x task-1-loss reduction at lambda_ewc=5)")


if __name__ == "__main__":
    run()
