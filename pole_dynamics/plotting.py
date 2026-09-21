"""Shared matplotlib helpers so every experiment's figures look consistent."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(fig, name):
    """Save a figure as PDF. `name` may be given with any extension (e.g.
    the historical "*.png") — it is always saved as PDF (vector format,
    resolution-independent), with the extension swapped automatically so
    none of the 39 experiment scripts need to change their calls."""
    base, _ext = os.path.splitext(name)
    pdf_name = base + ".pdf"
    path = os.path.join(OUTPUT_DIR, pdf_name)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  [figure saved] {path}")
    return path


def plot_complex_poles(ax, poles, label=None, color=None, marker="o", s=40):
    """Scatter complex poles with the unit circle drawn for reference."""
    poles = np.asarray(poles, dtype=complex)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.scatter(poles.real, poles.imag, label=label, color=color, marker=marker, s=s, zorder=5)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_aspect("equal", adjustable="box")
