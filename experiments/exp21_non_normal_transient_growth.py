"""
Experiment 21 -- vizexp label: viz:non-normal (Section 11.3, Remark 11.4)

Reproduces Script 25[C]: a highly non-normal matrix with small spectral
radius can still exhibit large transient growth (Trefethen-Embree
pseudospectral effect) before its eventual eigenvalue-governed decay
dominates -- the reason Prop. 11.2's operator-norm (singular-value) bound
is needed in addition to the eigenvalue picture.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics.plotting import savefig


def run():
    n = 6
    rho = 0.3
    # near-Jordan block: rho on the diagonal, large super-diagonal entries
    M = np.diag([rho] * n) + np.diag([8.0] * (n - 1), 1)
    eigs = np.linalg.eigvals(M)
    print("Experiment 21 (non-normal transient growth):")
    print(f"  spectral radius rho(M) = {np.max(np.abs(eigs)):.3f}")

    Ls = np.arange(1, 20)
    norms = []
    P = np.eye(n)
    for l in Ls:
        P = M @ P
        norms.append(np.linalg.norm(P, 2))
    norms = np.array(norms)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(Ls, norms, "o-", label="||M^L||_2 (actual)")
    ax.semilogy(Ls, rho ** Ls, "r--", label="rho(M)^L (eigenvalue-only prediction)")
    ax.set_xlabel("L"); ax.set_ylabel("operator norm")
    ax.set_title("Transient growth despite rho(M)<1 -- eigenvalues alone are unsafe")
    ax.legend()
    savefig(fig, "exp21_non_normal_transient_growth.png")
    print(f"  peak ||M^L||_2 = {norms.max():.1f}x initial norm at L={Ls[np.argmax(norms)]}")
    print("  (paper's Script 25[C]: peaks at 5965x around L=6)")


if __name__ == "__main__":
    run()
