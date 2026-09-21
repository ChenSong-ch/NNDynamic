"""
Experiment 11 -- vizexp label: viz:loss-poles (Section 6, Lemma 6.5, Table 1)

Verifies the per-loss output-space curvature Lambda(y) formulas of
Lemma 6.5 (Table "Lambda properties for common loss functions") by
finite-difference Hessians of each scalar loss w.r.t. its own logits,
compared against the closed-form expressions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import numerical_hessian
from pole_dynamics.plotting import savefig


def run():
    rng = np.random.default_rng(11)
    y_star = np.array([0.3])

    def mse(y):
        return 0.5 * np.sum((y - y_star) ** 2)

    def huber(y, delta=1.0):
        r = y[0] - y_star[0]
        return 0.5 * r ** 2 if abs(r) <= delta else delta * abs(r) - 0.5 * delta ** 2

    def logcosh(y):
        return np.log(np.cosh(y[0] - y_star[0]))

    def bce(y):
        sig = 1 / (1 + np.exp(-y[0]))
        return -(0.7 * np.log(sig + 1e-12) + 0.3 * np.log(1 - sig + 1e-12))

    y0 = np.array([0.1])
    results = {}
    results["MSE  (predicted Lambda=1)"] = numerical_hessian(mse, y0)[0, 0]
    r = 0.1 - y_star[0]
    huber_pred = 1.0 if abs(r) <= 1.0 else 0.0
    results[f"Huber (|r|<=delta, predicted {huber_pred})"] = numerical_hessian(huber, y0)[0, 0]
    logcosh_pred = 1 / np.cosh(0.1 - y_star[0]) ** 2
    results[f"log-cosh (predicted sech^2(r)={logcosh_pred:.4f})"] = numerical_hessian(logcosh, y0)[0, 0]
    sig0 = 1 / (1 + np.exp(-0.1))
    bce_pred = sig0 * (1 - sig0)
    results[f"BCE (predicted sigma(1-sigma)={bce_pred:.4f})"] = numerical_hessian(bce, y0)[0, 0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results.keys())
    vals = list(results.values())
    ax.barh(names, vals, color="C0")
    ax.set_xlabel("measured Lambda (finite-difference Hessian)")
    ax.set_title("Lemma 6.5: curvature Lambda of common losses matches closed form")
    savefig(fig, "exp11_loss_functions_poles.png")

    print("Experiment 11 (per-loss curvature Lambda):")
    for k, v in results.items():
        print(f"  {k}: measured = {v:.5f}")

    # Softmax cross-entropy: check Lambda @ 1_C = 0 (Corollary 6.9's structural null dir)
    def ce(y):
        z = y - y.max()
        p = np.exp(z) / np.exp(z).sum()
        return -np.log(p[0] + 1e-12)

    y0 = rng.normal(size=3) * 0.5
    H = numerical_hessian(ce, y0)
    print(f"  softmax-CE Lambda @ 1_C = {H @ np.ones(3)} (predicted: exactly 0)")


if __name__ == "__main__":
    run()
