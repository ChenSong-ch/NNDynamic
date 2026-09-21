"""
Experiment 23 -- vizexp label: viz:layer-gradients (Section 11.6, Prop. 11.10)

Verifies Prop. 11.10: sigma_min(D_{l->L}) >= (1-c)^(L-l-1) for a residual
network (no early layer's parameter block is arbitrarily attenuated),
vs. a plain network where it can be <= c^(L-l-1). We measure this using
random per-layer Jacobians exactly as in Experiment 20, but report the
per-layer downstream sigma_min as a function of depth-from-output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(23)
    n, L, c = 12, 60, 0.15
    Ms = []
    for _ in range(L):
        M = rng.normal(size=(n, n))
        M *= c / np.linalg.norm(M, 2)
        Ms.append(M)

    depth_from_output = []
    plain_sigmin, resnet_sigmin = [], []
    P_plain = np.eye(n)
    P_res = np.eye(n)
    plain_chain, res_chain = [], []
    for M in Ms:
        plain_chain.append(M)
        res_chain.append(np.eye(n) + M)

    for l in range(L):
        Dl = L - l - 1
        Pp = np.eye(n)
        Pr = np.eye(n)
        for M in plain_chain[l + 1:]:
            Pp = M @ Pp
        for Mr in res_chain[l + 1:]:
            Pr = Mr @ Pr
        depth_from_output.append(Dl)
        plain_sigmin.append(np.linalg.svd(Pp, compute_uv=False).min() if Dl > 0 else 1.0)
        resnet_sigmin.append(np.linalg.svd(Pr, compute_uv=False).min() if Dl > 0 else 1.0)

    depth_from_output = np.array(depth_from_output)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(depth_from_output, plain_sigmin, "o", ms=3, label="plain: sigma_min(D_{l->L})")
    ax.semilogy(depth_from_output, resnet_sigmin, "s", ms=3, label="residual: sigma_min(D_{l->L})")
    ax.semilogy(depth_from_output, (1 - c) ** depth_from_output, "r--", label="predicted floor (1-c)^(L-l-1)")
    ax.set_xlabel("L - l - 1 (downstream depth)")
    ax.set_ylabel("sigma_min(D_l->L)")
    ax.set_title("Proposition 11.10: no early residual layer is arbitrarily starved")
    ax.legend()
    savefig(fig, "exp23_layer_gradient_norms.png")
    print("Experiment 23 (per-layer parameter-block conditioning): see figure.")


if __name__ == "__main__":
    run()
