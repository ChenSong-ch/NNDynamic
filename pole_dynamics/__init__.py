"""
pole_dynamics
=============
Shared numerical toolkit for the experiments in this repository.

Every experiment in ``experiments/`` reproduces (at small, laptop-friendly
scale) one ``\\begin{vizexp}...\\end{vizexp}`` block from
"Neural Network Training as a Dynamical System". Scale is reduced
following exactly the same convention the paper's own Appendix A
("Numerical Verification", Scripts 1-38) already uses: plain NumPy/SciPy,
small synthetic or scikit-learn-bundled datasets, no GPU, no internet
downloads. Where the paper's vizexp calls for a large pretrained model
(ResNet-50/ImageNet, GPT-2, ViT-L, StyleGAN2, DCGAN/MNIST-GAN) we use a
structurally faithful small proxy (a 2-4 layer MLP/CNN or a closed-form
toy) that exhibits the *same provable mechanism* the proposition claims,
and we say so explicitly in the experiment's docstring and in
``README.md``.

Nothing here is a black box: Jacobians, Hessians and Hessian-vector
products are computed either in closed form or by finite differences,
so every number an experiment prints can be checked by hand against the
paper's equations.
"""

from .core import (
    TinyMLP,
    finite_diff_hvp,
    hvp_generic,
    numerical_gradient,
    numerical_jacobian,
    numerical_hessian,
    lanczos_topk,
    gd_poles,
    set_seed,
)
from .plotting import (
    plot_complex_poles,
    savefig,
)

__all__ = [
    "TinyMLP",
    "finite_diff_hvp",
    "hvp_generic",
    "numerical_gradient",
    "numerical_jacobian",
    "numerical_hessian",
    "lanczos_topk",
    "gd_poles",
    "set_seed",
    "plot_complex_poles",
    "savefig",
]
