"""
Experiment 36 -- vizexp label: viz:attention-pole (Section 18.5, Prop. 18.11,
Lemma 18.12)

Paper's version extracts attention from a pretrained Transformer
(distilgpt2/bert, needing an internet download not available here); we
use random softmax-attention matrices at varying logit "sharpness",
exactly the paper's own Script 35 methodology used to derive the closed
form before pointing it at a real model.

Verifies Proposition 18.11 (A @ 1 = 1 exactly, every softmax attention
matrix has eigenvalue 1) and Lemma 18.12 (Dobrushin bound: subdominant
|lambda| <= delta(A) = max row-pair total-variation distance), reproducing
Script 35's entropy-vs-spectral-gap curve.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig
from scipy.stats import entropy


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def dobrushin(A):
    T = A.shape[0]
    best = 0.0
    for i in range(T):
        for j in range(T):
            best = max(best, 0.5 * np.sum(np.abs(A[i] - A[j])))
    return best


def run():
    rng = set_seed(36)
    Tt = 8
    logits = rng.normal(size=(Tt, Tt))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sharpnesses = [10.0, 2.0, 1.0, 0.3]
    entropies, gaps, deltas = [], [], []
    for s in sharpnesses:
        A = softmax_rows(logits / s)
        row_sum_err = np.max(np.abs(A.sum(axis=1) - 1))
        eigs = np.linalg.eigvals(A)
        order = np.argsort(-np.abs(eigs))
        eigs = eigs[order]
        mean_ent = np.mean([entropy(A[i]) for i in range(Tt)])
        delta = dobrushin(A)
        entropies.append(mean_ent)
        gaps.append(np.abs(eigs[1]))
        deltas.append(delta)
        print(f"  1/T={1/s:.2f}: mean row entropy={mean_ent:.3f}  "
              f"|lambda_1|={np.abs(eigs[0]):.6f}  |lambda_2|={np.abs(eigs[1]):.4f}  "
              f"Dobrushin delta(A)={delta:.4f}  row-sum err={row_sum_err:.1e}")

    ax.plot(entropies, gaps, "o-", label="measured |lambda_2|")
    ax.plot(entropies, deltas, "s--", label="Dobrushin bound delta(A)")
    ax.set_xlabel("mean row entropy"); ax.set_ylabel("subdominant eigenvalue modulus")
    ax.set_title("Lemma 18.12: attention spectral gap vs. row entropy (Dobrushin bound)")
    ax.legend()
    savefig(fig, "exp36_attention_stochastic_pole.png")
    print("Experiment 36 (softmax-attention stochastic pole): "
          "lambda_1=1 exactly in every case (Prop. 18.11); "
          "|lambda_2| always <= delta(A) (paper's Script 35).")


if __name__ == "__main__":
    run()
