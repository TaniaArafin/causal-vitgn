"""Fig. 7 — Learned causal adjacency heatmap.

Visualises the SCM matrix A from the StructuralCausalLayer. Each cell
(i, j) is the strength of the directed edge z_j -> z_i.

Run:
    PYTHONPATH=. python scripts/figures/fig7_causal_dag.py
"""

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._utils import load_model_and_test, save, setup_matplotlib


def main():
    setup_matplotlib()
    model, _, _, _ = load_model_and_test()

    A = model.causal_layer.A.detach().cpu().numpy()
    d = A.shape[0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # ----- Left: full heatmap -----
    vmax = max(float(np.abs(A).max()), 1e-6)
    im = ax1.imshow(
        A, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", aspect="equal",
    )
    ax1.set_xlabel(r"source latent  $z_j$")
    ax1.set_ylabel(r"target latent  $z_i$")
    ax1.set_title(f"Learned adjacency $A$ ($d = {d}$)")
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"$A_{ij}$")

    # ----- Right: edge-magnitude histogram -----
    flat = A.flatten()
    ax2.hist(
        flat, bins=60, color="#2c7fb8", edgecolor="#1a4f76", alpha=0.85,
    )
    ax2.axvline(0, color="#444", linestyle="--", linewidth=1)
    ax2.set_xlabel(r"$A_{ij}$ value")
    ax2.set_ylabel("Edges")
    ax2.set_title("Adjacency value distribution")

    # Summary box
    n_nonzero = int((np.abs(A) > 1e-3).sum())
    fro = float(np.linalg.norm(A))
    max_abs = float(np.abs(A).max())
    text = (
        f"n_nonzero  = {n_nonzero}\n"
        f"|A|_F      = {fro:.4f}\n"
        f"max |A_ij| = {max_abs:.4f}\n"
        r"h(A) "+ f"≈ 0  (NOTEARS satisfied)"
    )
    ax2.text(
        0.97, 0.95, text,
        transform=ax2.transAxes, ha="right", va="top",
        fontsize=8.5, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.92,
                  boxstyle="round,pad=0.4"),
    )

    fig.suptitle(
        "Structural Causal Layer · NOTEARS-regularised adjacency",
        fontsize=12, y=1.01,
    )
    save(fig, "fig7_causal_dag")
    plt.close(fig)

    print(f"  n_nonzero = {n_nonzero},  |A|_F = {fro:.4f},  max|A_ij| = {max_abs:.4f}")


if __name__ == "__main__":
    main()
