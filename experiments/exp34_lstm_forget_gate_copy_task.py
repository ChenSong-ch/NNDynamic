"""
Experiment 34 -- vizexp label: viz:lstm-copy (Section 18.3.1, Cor. 18.8)

Paper: forget-gate poles on the Hochreiter/Schmidhuber copy task, vanilla
RNN vs. LSTM. Proxy at small scale (hidden size 8, copy length up to 60).
Verifies Corollary 18.8: an LSTM with forget-gate bias initialised
positive can hold information across arbitrarily long delays (gates
saturate near 1, cell-state pole near z=1) while a vanilla RNN's
attenuation rho(M)^T collapses for large T.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run():
    rng = set_seed(34)
    H = 8

    # Vanilla RNN attenuation: rho(W_h)^T with a contractive W_h (rho=0.85)
    Wh = rng.normal(size=(H, H))
    Wh *= 0.85 / np.max(np.abs(np.linalg.eigvals(Wh)))
    rho = np.max(np.abs(np.linalg.eigvals(Wh)))

    # LSTM: forget gate initialised with positive bias b_f so f_t = sigmoid(Wf@h + b_f) ~ 1
    Wf = rng.normal(size=(H, H)) * 0.1
    b_f = 3.0  # standard "forget-gate bias" init (Jozefowicz et al., 2015)

    Ts = np.arange(1, 80, 2)
    rnn_atten = rho ** Ts

    # simulate an LSTM holding a fixed cell state through T blank steps, track prod(f_t)
    h = np.zeros(H)
    c = rng.normal(size=H) * 0.5
    retained = []
    x_blank = np.zeros(H)
    prod_f = 1.0
    prod_f_hist = []
    for t in range(1, Ts.max() + 1):
        f_t = sigmoid(Wf @ h + b_f)
        i_t = sigmoid(rng.normal(size=H) * 0.1)  # input gate, small/irrelevant during blanks
        c = f_t * c  # blank input: no new information added (i_t*ctilde ~ 0 during copy delay)
        h = np.tanh(c) * sigmoid(rng.normal(size=H) * 0.1)
        prod_f *= np.mean(f_t)
        prod_f_hist.append(prod_f)
    prod_f_hist = np.array(prod_f_hist)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(Ts, rnn_atten, "o-", label=f"vanilla RNN: rho(W_h)^T, rho={rho:.2f}")
    ax.semilogy(np.arange(1, Ts.max() + 1), prod_f_hist, "s-", ms=3,
               label="LSTM: product of mean forget gate over delay")
    ax.set_xlabel("delay length T"); ax.set_ylabel("retention (attenuation factor)")
    ax.set_title("Corollary 18.8: LSTM forget gate vs. vanilla RNN attenuation")
    ax.legend()
    savefig(fig, "exp34_lstm_forget_gate_copy_task.png")

    print("Experiment 34 (LSTM forget-gate vs. vanilla RNN retention):")
    print(f"  vanilla RNN at T=79: retention = {rnn_atten[-1]:.3e}")
    print(f"  LSTM at T=79: retention = {prod_f_hist[-1]:.4f} (forget-gate bias={b_f} -> near-1 gates)")


if __name__ == "__main__":
    run()
