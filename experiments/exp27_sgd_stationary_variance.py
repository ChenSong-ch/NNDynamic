"""
Experiment 27 -- vizexp label: viz:sgd-variance (Section 14, Theorem 14.2)

Reproduces Script 23[B]: stationary covariance of linearised SGD near a
fixed point solves the discrete Lyapunov equation Sigma=A Sigma A^T +
eta^2*C; in the common-eigenbasis case the modal solution is
eta*c_i/(lambda_i*(2-eta*lambda_i)), diverging both for flat directions
(lambda_i->0) and near the Edge of Stability (eta*lambda_i->2).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(27)
    lambdas = np.array([2.0, 0.5, 0.05])
    eta = 0.4
    c = np.array([0.01, 0.01, 0.01])
    steps = 200_000

    e = np.zeros(3)
    hist = np.zeros((steps, 3))
    for k in range(steps):
        xi = rng.normal(size=3) * np.sqrt(c)
        e = (1 - eta * lambdas) * e - eta * xi
        hist[k] = e

    burn = steps // 5
    empirical_var = hist[burn:].var(axis=0)
    predicted_var = eta * c / (lambdas * (2 - eta * lambdas))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = [f"lambda={l}" for l in lambdas]
    x = np.arange(3)
    ax.bar(x - 0.15, empirical_var, width=0.3, label="empirical stationary variance")
    ax.bar(x + 0.15, predicted_var, width=0.3, label="predicted eta*c/(lambda*(2-eta*lambda))")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("stationary variance")
    ax.set_title("Theorem 14.2: SGD stationary covariance (Lyapunov equation)")
    ax.legend()
    savefig(fig, "exp27_sgd_stationary_variance.png")

    print("Experiment 27 (SGD stationary Lyapunov covariance):")
    print(f"  empirical:  {empirical_var}")
    print(f"  predicted:  {predicted_var}")
    print("  (paper's Script 23[B]: empirical 0.00166 0.00441 0.03999; "
          "predicted 0.00167 0.00444 0.04040)")


if __name__ == "__main__":
    run()
