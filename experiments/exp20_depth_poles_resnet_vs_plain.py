"""
Experiment 20 -- vizexp label: viz:depth-poles (Section 11, Prop. 11.2)

Verifies Proposition 11.2 (residual depth-propagation floor):
  sigma_min(I+J) >= 1-c  vs.  sigma_min(prod J_l) <= c^L -> 0.
Random per-layer Jacobians of operator norm ~c, depth up to L=200 layers
(exactly the paper's own Script 9 setup).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import set_seed
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(20)
    n, L, c = 20, 200, 0.1
    plain_min, resnet_min, depths = [], [], []
    P_plain = np.eye(n)
    P_resnet = np.eye(n)
    for l in range(L):
        M = rng.normal(size=(n, n))
        M *= c / np.linalg.norm(M, 2)
        P_plain = M @ P_plain
        P_resnet = (np.eye(n) + M) @ P_resnet
        if l % 5 == 0 or l == L - 1:
            depths.append(l)
            plain_min.append(np.linalg.svd(P_plain, compute_uv=False).min())
            resnet_min.append(np.linalg.svd(P_resnet, compute_uv=False).min())

    depths = np.array(depths)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(depths, np.array(plain_min) + 1e-300, "o-", label="plain feedforward sigma_min")
    ax.semilogy(depths, np.array(resnet_min) + 1e-300, "s-", label="residual sigma_min")
    ax.semilogy(depths, (1 - c) ** depths, "r--", label="predicted floor (1-c)^L")
    ax.set_xlabel("depth L"); ax.set_ylabel("sigma_min of cumulative product")
    ax.set_title("Proposition 11.2: residual floor vs. plain-network collapse")
    ax.legend()
    savefig(fig, "exp20_depth_poles_resnet_vs_plain.png")

    print("Experiment 20 (depth-propagation floor, plain vs residual):")
    print(f"  at depth {depths[-1]}: plain sigma_min={plain_min[-1]:.3e}, "
          f"resnet sigma_min={resnet_min[-1]:.3e}, predicted floor={(1-c)**depths[-1]:.3e}")


if __name__ == "__main__":
    run()
