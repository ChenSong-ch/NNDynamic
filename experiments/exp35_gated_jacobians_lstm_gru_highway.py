"""
Experiment 35 -- vizexp label: viz:gated-jacobians (Section 18.3.1-18.4,
Prop. 18.6, Remark 18.9, Prop. 18.10)

Reproduces Scripts 27[A,B] and 30[A]: LSTM's cell-to-cell Jacobian block
d c_t/d c_{t-1} equals diag(f_t) EXACTLY (finite-difference precision);
GRU's h-to-h Jacobian has a substantial dense correction beyond
diag(1-z_t); Highway's depth Jacobian is similarly dense (~50% of the
Frobenius norm is NOT explained by the gate diagonal alone).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed, numerical_jacobian
from pole_dynamics.plotting import savefig


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run():
    rng = set_seed(35)
    H = 4
    Wf, Wi, Wc = [rng.normal(size=(H, 2 * H)) * 0.5 for _ in range(3)]
    Wz, Wr, Wh_ = [rng.normal(size=(H, 2 * H)) * 0.5 for _ in range(3)]
    WT, WHh = [rng.normal(size=(H, H)) * 0.5 for _ in range(2)]
    bT = -0.2

    h0 = rng.normal(size=H) * 0.3
    c0 = rng.normal(size=H) * 0.3
    x = rng.normal(size=H) * 0.3

    # ---- LSTM ----
    def lstm_step(hc):
        h, c = hc[:H], hc[H:]
        hx = np.concatenate([h, x])
        f = sigmoid(Wf @ hx)
        i = sigmoid(Wi @ hx)
        ctil = np.tanh(Wc @ hx)
        c_new = f * c + i * ctil
        h_new = np.tanh(c_new) * sigmoid(rng.normal(size=H) * 0.01)  # output gate proxy (fixed noise seed issue below)
        return np.concatenate([h_new, c_new])

    hc0 = np.concatenate([h0, c0])
    J_lstm = numerical_jacobian(lstm_step, hc0)
    dc_dc = J_lstm[H:, H:]
    hx0 = np.concatenate([h0, x])
    f0 = sigmoid(Wf @ hx0)
    resid_lstm = np.linalg.norm(dc_dc - np.diag(f0))
    print("Experiment 35 (LSTM/GRU/Highway gated Jacobians):")
    print(f"  LSTM: ||dc_t/dc_{{t-1}} - diag(f_t)|| = {resid_lstm:.3e} (predicted: exactly 0)")

    # ---- GRU ----
    def gru_step(h):
        hx = np.concatenate([h, x])
        z = sigmoid(Wz @ hx)
        r = sigmoid(Wr @ hx)
        hx_r = np.concatenate([r * h, x])
        htil = np.tanh(Wh_ @ hx_r)
        return (1 - z) * h + z * htil

    J_gru = numerical_jacobian(gru_step, h0)
    hx0 = np.concatenate([h0, x])
    z0 = sigmoid(Wz @ hx0)
    diag_part = np.diag(1 - z0)
    resid_gru = np.linalg.norm(J_gru - diag_part) / np.linalg.norm(J_gru)
    print(f"  GRU: ||J_full - diag(1-z_t)|| / ||J_full|| = {resid_gru:.3f} "
          f"(predicted: NOT negligible, unlike LSTM)")

    # ---- Highway ----
    def highway_step(h):
        T = sigmoid(WT @ h + bT)
        Hh = np.tanh(WHh @ h)
        return T * Hh + (1 - T) * h

    J_hw = numerical_jacobian(highway_step, h0)
    T0 = sigmoid(WT @ h0 + bT)
    diag_hw = np.diag(1 - T0)
    frac_explained = np.linalg.norm(diag_hw) ** 2 / np.linalg.norm(J_hw) ** 2
    print(f"  Highway: ||diag(1-T)||_F^2 / ||J_full||_F^2 = {frac_explained:.3f} "
          f"(paper's Script 30[A]: 0.48 -- about half, dense correction NOT negligible)")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(["LSTM\n(exact channel)", "GRU\n(dense correction)", "Highway\n(dense correction)"],
           [resid_lstm, resid_gru, 1 - frac_explained], color=["C2", "C1", "C0"])
    ax.set_ylabel("relative residual beyond the diagonal gate channel")
    ax.set_title("Prop. 18.6 / Remark 18.9 / Prop. 18.10: only LSTM has an exact clean channel")
    savefig(fig, "exp35_gated_jacobians_lstm_gru_highway.png")


if __name__ == "__main__":
    run()
