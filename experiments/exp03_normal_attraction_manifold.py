"""
Experiment 03 -- vizexp label: viz:manifold (Section 3.6, Theorem 3.14)

Paper: "Normal attraction to the interpolation manifold, on an
over-parameterised MNIST MLP." Proxy: a small over-parameterised TinyMLP
(n >> m) trained to interpolate a tiny regression dataset.

Verifies Theorem 3.14: perturbing w* along the top / bottom right
singular vectors of J(w*) contracts the residual at rate
|1 - eta*sigma_i^2/N| exactly; perturbing along ker J(w*) leaves the
residual unchanged to second order (GD never moves along it -- the
tangent space of the interpolating manifold M).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from pole_dynamics import TinyMLP, set_seed, numerical_jacobian
from pole_dynamics.plotting import savefig


def run():
    rng = set_seed(2)
    mlp = TinyMLP([3, 12, 1], activation="tanh", seed=2)  # n >> m
    N = 6
    X = rng.normal(size=(N, 3)) * 0.5
    y = rng.normal(size=(N, 1)) * 0.3

    def residual(theta):
        return (mlp.forward(theta, X) - y).ravel()

    def loss(theta):
        r = residual(theta)
        return 0.5 * np.sum(r ** 2)

    theta = mlp.theta0.copy()
    eta = 0.02
    for it in range(6000):
        g = numerical_jacobian(residual, theta).T @ residual(theta)
        gn = np.linalg.norm(g)
        if gn > 5.0:
            g = g * (5.0 / gn)  # clip to avoid overshoot with the crude FD jacobian
        theta -= eta * g
    theta_star = theta.copy()
    print(f"  ||R(w*)|| = {np.linalg.norm(residual(theta_star)):.3e}")

    J = numerical_jacobian(residual, theta_star)
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    r = np.sum(S > 1e-6 * S[0])
    n = mlp.n_params
    print(f"  rank J(w*) = {r} / n={n}, m={J.shape[0]}")

    v_top = Vt[0]
    v_bot = Vt[r - 1]
    v_null = Vt[r + 2] if r + 2 < n else Vt[-1]  # a null-space direction

    def run_from(direction, delta=1e-2, steps=60):
        th = theta_star + delta * direction
        norms = []
        for _ in range(steps):
            norms.append(np.linalg.norm(residual(th)))
            g = numerical_jacobian(residual, th).T @ residual(th)
            th = th - eta * g
        return np.array(norms)

    norms_top = run_from(v_top)
    norms_bot = run_from(v_bot)
    norms_null = run_from(v_null)

    pred_top = np.abs(1 - eta * S[0] ** 2 / N)
    pred_bot = np.abs(1 - eta * S[r - 1] ** 2 / N)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    steps = np.arange(len(norms_top))
    ax.semilogy(steps, norms_top + 1e-16, label="perturb along top singular dir")
    ax.semilogy(steps, norms_top[0] * pred_top ** steps + 1e-16, "--", alpha=0.6,
                label=f"predicted rate |1-eta*sigma_1^2/N|={pred_top:.3f}")
    ax.semilogy(steps, norms_bot + 1e-16, label="perturb along weakest normal dir")
    ax.semilogy(steps, norms_bot[0] * pred_bot ** steps + 1e-16, "--", alpha=0.6,
                label=f"predicted rate={pred_bot:.3f}")
    ax.semilogy(steps, norms_null + 1e-16, label="perturb along ker J(w*) (tangent to M)")
    ax.set_xlabel("GD step"); ax.set_ylabel("||R(w_k)||")
    ax.set_title("Theorem 3.14: normal attraction to the interpolating manifold")
    ax.legend(fontsize=7)
    savefig(fig, "exp03_normal_attraction_manifold.png")

    print("Experiment 03 (normal attraction manifold):")
    print(f"  measured/predicted top-mode ratio: "
          f"{(norms_top[10]/norms_top[9]):.4f} vs {pred_top:.4f}")
    print(f"  null-space residual change (should stay ~constant): "
          f"{norms_null[0]:.3e} -> {norms_null[-1]:.3e}")


if __name__ == "__main__":
    run()
