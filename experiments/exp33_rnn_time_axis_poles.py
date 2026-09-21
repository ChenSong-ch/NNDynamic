"""
Experiment 33 -- vizexp label: viz:rnn-ptb (Section 18.3, Theorem 18.4)

Paper's version trains a character RNN on Penn Treebank; proxy: a
vanilla tanh RNN on a small synthetic sequence, at three initial
spectral-radius scalings of W_h. Verifies Theorem 18.4(2): the gradient
norm ||d h_T/d h_t||_2 is bounded by (sigma_bar*||W_h||_2)^(T-t), and
this bound is broadband (Corollary of Prop. 18.1) -- governed only by
the operator norm, independent of phase/frequency.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(33)
    H = 16
    T = 40
    results = {}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for radius in [0.5, 1.0, 1.5]:
        Wh = rng.normal(size=(H, H))
        Wh *= radius / np.linalg.norm(Wh, 2)  # scale OPERATOR norm to match Theorem 18.4's ||W_h||_2 hypothesis
        x = rng.normal(size=(T, H)) * 0.1
        h = np.zeros(H)
        traj = [h.copy()]
        for t in range(T):
            h = np.tanh(Wh @ h + x[t])
            traj.append(h.copy())

        # linearised Jacobians M_t = diag(1-h_{t+1}^2) @ W_h
        norms = []
        P = np.eye(H)
        for t in range(T - 1, -1, -1):
            Mt = np.diag(1 - traj[t + 1] ** 2) @ Wh
            P = P @ Mt
            norms.append(np.linalg.norm(P, 2))
        norms = norms  # norms[i] already corresponds to T-t = i+1 (no reversal needed)
        Tminus_t = np.arange(1, T + 1)
        ax.semilogy(Tminus_t, norms, label=f"measured, init spectral radius={radius}")
        bound = (radius) ** Tminus_t  # sigma_bar<=1 for tanh, ||W_h||~radius here
        ax.semilogy(Tminus_t, bound, "--", alpha=0.5, color=ax.lines[-1].get_color(),
                    label=f"bound (radius)^(T-t), r={radius}")
        results[radius] = (norms[-1], bound[-1])

    ax.set_xlabel("T - t"); ax.set_ylabel("||d h_T / d h_t||_2")
    ax.set_title("Theorem 18.4: vanilla RNN gradient norm vs. depth (time axis)")
    ax.legend(fontsize=7)
    savefig(fig, "exp33_rnn_time_axis_poles.png")

    print("Experiment 33 (RNN time-axis vanishing/exploding gradient):")
    for radius, (measured, bound) in results.items():
        print(f"  radius={radius}: ||dh_T/dh_0||={measured:.3e}  operator-norm bound={bound:.3e}")


if __name__ == "__main__":
    run()
