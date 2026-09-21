"""
Experiment 16 -- vizexp label: viz:gen-gap (Section 10.2, Prop. 10.3)

Paper: trajectory-gap N^{-1/2}/mu law between two GD runs on disjoint
i.i.d. data subsets, MNIST-logistic-regression proxy via sklearn digits.
Verifies Proposition 10.3: ||delta_k|| = O_p(N^{-1/2}/mu).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run():
    data = load_digits(n_class=2)
    Xall = data.data / 16.0
    yall = data.target.astype(float)
    Xall = np.hstack([Xall, np.ones((Xall.shape[0], 1))])
    d = Xall.shape[1]

    Ns = [20, 40, 80, 150]
    lam_reg = 0.05
    eta = 0.5

    rng = set_seed(16)
    gaps = []
    for N in Ns:
        pair_gaps = []
        for trial in range(5):
            idx = rng.choice(len(yall), size=2 * N, replace=False)
            X1, y1 = Xall[idx[:N]], yall[idx[:N]]
            X2, y2 = Xall[idx[N:]], yall[idx[N:]]
            w1 = np.zeros(d); w2 = np.zeros(d)
            for _ in range(300):
                g1 = X1.T @ (sigmoid(X1 @ w1) - y1) / N + lam_reg * w1
                g2 = X2.T @ (sigmoid(X2 @ w2) - y2) / N + lam_reg * w2
                w1 -= eta * g1
                w2 -= eta * g2
            pair_gaps.append(np.linalg.norm(w1 - w2))
        gaps.append(np.mean(pair_gaps))
    gaps = np.array(gaps)

    logN = np.log(Ns)
    logGap = np.log(gaps)
    slope, intercept = np.polyfit(logN, logGap, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(Ns, gaps, "o-", label="measured ||w_N^(1)-w_N^(2)||")
    ax.loglog(Ns, np.exp(intercept) * np.array(Ns, dtype=float) ** (-0.5), "--",
              label="reference slope -1/2")
    ax.set_xlabel("N (samples per split)"); ax.set_ylabel("trajectory separation")
    ax.set_title(f"Proposition 10.3: N^-1/2 gap law (fit slope={slope:.3f})")
    ax.legend()
    savefig(fig, "exp16_generalization_gap.png")

    print("Experiment 16 (generalisation-gap N^-1/2 scaling):")
    print(f"  fitted slope = {slope:.3f} (predicted approx -0.5)")


if __name__ == "__main__":
    run()
