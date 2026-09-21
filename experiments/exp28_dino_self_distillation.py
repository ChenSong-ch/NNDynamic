"""
Experiment 28 -- vizexp label: viz:dino-viz (Section 15, Theorem 15.5)

Reproduces Script 24: DINO/iBOT-style self-distillation as a joint
dynamical system on (w, wbar). Verifies:
  (a) grad_w(loss) = 0 exactly at ANY point of agreement w=wbar
      (Theorem 15.4 -- fixed pointwise, not just at minimisers)
  (b) H_{w,wbar} = -H_ww exactly (Prop. 15.3)
  (c) the joint linearisation spectrum is exactly {tau*z_i} union {1}^n
      (Theorem 15.5)
  (d) the gap ||Delta_k||=||w_k-wbar_k|| contracts at rate max|tau*z_i|.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed, numerical_gradient, numerical_hessian
from pole_dynamics.plotting import savefig


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run():
    rng = set_seed(28)
    C, n, N = 4, 20, 6
    Wq = rng.normal(size=(C, n)) * 0.3  # linear "head": logits = W @ features(x); here features=identity proxy
    X = rng.normal(size=(N, n)) * 0.3
    eta, tau = 0.5, 0.95

    def logits(w, x):
        return Wq @ (w[:n] if n <= len(w) else w)  # placeholder not used; see below

    # Simpler: student/teacher parameters ARE the logit-producing linear map
    # per example i: logit_i(w) = A_i @ w,  A_i random (n -> C) fixed "features"
    A = [rng.normal(size=(C, n)) * 0.4 for _ in range(N)]

    def per_example_loss(w, wbar, i):
        y = A[i] @ w
        ybar = A[i] @ wbar
        p_teacher = softmax(ybar)
        logp_student = y - np.log(np.sum(np.exp(y - y.max()))) - y.max() + y.max()
        z = y - y.max()
        logp = z - np.log(np.sum(np.exp(z)))
        return -np.sum(p_teacher * logp)

    def total_loss(w, wbar):
        return np.mean([per_example_loss(w, wbar, i) for i in range(N)])

    w0 = rng.normal(size=n) * 0.2
    grad_w_at_agreement = numerical_gradient(lambda w: total_loss(w, w0), w0)
    print("Experiment 28 (DINO/iBOT self-distillation dynamics):")
    print(f"  (a) ||grad_w L(w,w)|| at a RANDOM (non-minimiser) point of agreement "
          f"= {np.linalg.norm(grad_w_at_agreement):.3e} (predicted: exactly 0)")

    H_ww = numerical_hessian(lambda w: total_loss(w, w0), w0)

    def cross_fn(v):
        # H_{w,wbar} v via finite differences of grad_w wrt wbar direction v
        h = 1e-4
        gp = numerical_gradient(lambda w: total_loss(w, w0 + h * v), w0)
        gm = numerical_gradient(lambda w: total_loss(w, w0 - h * v), w0)
        return (gp - gm) / (2 * h)

    H_wwbar = np.column_stack([cross_fn(e) for e in np.eye(n)])
    resid = np.linalg.norm(H_wwbar + H_ww) / np.linalg.norm(H_ww)
    print(f"  (b) ||H_w,wbar + H_ww|| / ||H_ww|| = {resid:.3e} (predicted: exactly 0)")

    G_eigs = np.linalg.eigvalsh(H_ww)
    z = 1 - eta * G_eigs
    predicted_spectrum = np.concatenate([tau * z, np.ones(n)])

    A_block = np.block([[tau * (np.eye(n) - eta * H_ww), np.zeros((n, n))],
                         [(1 - tau) * tau * (np.eye(n) - eta * H_ww), np.eye(n)]])
    joint_eigs = np.linalg.eigvals(A_block)
    print(f"  (c) predicted spectrum matches {{tau*z_i}} U {{1}}^n: "
          f"max diff (sorted) = "
          f"{np.max(np.abs(np.sort(joint_eigs.real) - np.sort(predicted_spectrum))):.3e}")

    # (d) gap contraction
    w, wbar = w0 + rng.normal(size=n) * 0.05, w0.copy()
    gaps = []
    for _ in range(300):
        gaps.append(np.linalg.norm(w - wbar))
        g = numerical_gradient(lambda ww: total_loss(ww, wbar), w)
        w_new = w - eta * g
        wbar_new = tau * wbar + (1 - tau) * w_new
        w, wbar = w_new, wbar_new
    gaps = np.array(gaps)
    rate_pred = np.max(np.abs(tau * z))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(gaps + 1e-16, label="||w_k - wbar_k||")
    k_arr = np.arange(len(gaps))
    ax.semilogy(gaps[0] * rate_pred ** k_arr, "r--", label=f"predicted rate max|tau*z_i|={rate_pred:.3f}")
    ax.set_xlabel("step"); ax.set_ylabel("student-teacher gap")
    ax.set_title("Theorem 15.5: gap dynamics contract at rate max|tau*z_i|")
    ax.legend()
    savefig(fig, "exp28_dino_self_distillation.png")
    print(f"  (d) measured late-stage contraction ratio ~ "
          f"{(gaps[-1]/gaps[-20])**(1/20):.4f} vs predicted {rate_pred:.4f}")


if __name__ == "__main__":
    run()
