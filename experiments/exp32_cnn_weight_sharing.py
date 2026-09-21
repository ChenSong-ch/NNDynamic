"""
Experiment 32 -- vizexp label: viz:weight-sharing (Section 18.2, Prop. 18.2)

Reproduces Script 27[D]: weight sharing aggregates curvature across
spatial location. lambda_k = ||sum_p j_p||_Lambda^2 / N vs. the average
UNTIED curvature lambda_bar; verifies 0 <= lambda_k <= P^2*lambda_bar,
with the ceiling approached when per-location gradients are aligned and
lambda_k/lambda_bar ~ O(1) for random rows, ~1 for alternating-sign rows.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(32)
    P, d_out = 5, 3

    def boost(rows):
        rows = np.array(rows)
        shared = np.linalg.norm(rows.sum(axis=0)) ** 2 / P
        avg_unshared = np.mean([np.linalg.norm(r) ** 2 for r in rows]) / P
        return shared, avg_unshared, shared / avg_unshared

    base = rng.normal(size=d_out)
    aligned_rows = [base * (1 + 0.1 * rng.normal()) for _ in range(P)]
    random_rows = [rng.normal(size=d_out) for _ in range(P)]
    alt_rows = [base * ((-1) ** i) for i in range(P)]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    names, ratios = [], []
    for name, rows in [("aligned", aligned_rows), ("random", random_rows), ("alternating", alt_rows)]:
        shared, avg, ratio = boost(rows)
        names.append(name); ratios.append(ratio)
        print(f"  {name:12s}: shared={shared:.3f}  avg_unshared={avg:.3f}  boost={ratio:.2f}x "
              f"(ceiling P^2={P**2})")
    ax.bar(names, ratios, color=["C0", "C1", "C2"])
    ax.axhline(P ** 2, color="r", ls="--", label=f"Cauchy-Schwarz ceiling P^2={P**2}")
    ax.axhline(1.0, color="k", ls=":", label="no aggregation benefit (ratio=1)")
    ax.set_ylabel("curvature boost lambda_k / lambda_bar")
    ax.set_title("Proposition 18.2: weight-sharing curvature multiplier")
    ax.legend()
    savefig(fig, "exp32_cnn_weight_sharing.png")
    print("Experiment 32 (CNN weight-sharing curvature aggregation): see figure "
          "(paper's Script 27[D]: aligned 24.8x, random 3.16x, alternating 1.00x).")


if __name__ == "__main__":
    run()
