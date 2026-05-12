"""Fig. 4 — Posterior uncertainty vs. cascade virality.

For each test cascade, plot:
  x = cascade size (proxy for virality)
  y = entropy of the variational posterior q(z|h,c)

Hypothesis: larger cascades have more deterministic propagation, so the
encoder should be more confident (lower posterior variance/entropy).

Run:
    PYTHONPATH=. python scripts/figures/fig4_uncertainty_virality.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.figures._utils import load_model_and_test, save, setup_matplotlib
from src.utils.data_loader import collate_cascades


def gaussian_entropy(logvar: torch.Tensor) -> torch.Tensor:
    """H[N(mu, sigma^2)] = 0.5 * sum(log(2*pi*e) + logvar) per sample."""
    return 0.5 * (np.log(2 * np.pi * np.e) * logvar.shape[-1] + logvar.sum(dim=-1))


def main():
    setup_matplotlib()
    model, _, test_ds, device = load_model_and_test()

    loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        collate_fn=collate_cascades,
    )

    sizes = []   # number of non-pad neighbours
    entropies = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            out = model(batch_dev)
            ent = gaussian_entropy(out["logvar"]).cpu().numpy()

            nb = batch["neighbor_nodes"]  # (B, K)
            sz = (nb != 0).sum(dim=-1).cpu().numpy()

            sizes.extend(sz.tolist())
            entropies.extend(ent.tolist())

    sizes = np.array(sizes)
    entropies = np.array(entropies)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(sizes, entropies, s=10, alpha=0.25, color="#2c7fb8",
               edgecolors="none")

    # Add a binned mean trend line
    bins = np.linspace(sizes.min(), sizes.max(), 12)
    means, centers = [], []
    for i in range(len(bins) - 1):
        m = (sizes >= bins[i]) & (sizes < bins[i + 1])
        if m.sum() >= 5:
            means.append(entropies[m].mean())
            centers.append(0.5 * (bins[i] + bins[i + 1]))
    if centers:
        ax.plot(centers, means, "o-", color="#d62728", linewidth=2,
                markersize=6, label="binned mean")

    ax.set_xlabel("Cascade size (observed retweeters)")
    ax.set_ylabel(r"Posterior entropy  $H[q(z|h,c)]$")
    ax.set_title("Posterior uncertainty decreases for larger cascades")
    ax.legend(loc="best", framealpha=0.9)

    # Correlation in the caption
    corr = np.corrcoef(sizes, entropies)[0, 1]
    ax.text(
        0.97, 0.05,
        f"Pearson r = {corr:+.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.35"),
        fontsize=9,
    )

    save(fig, "fig4_uncertainty_virality")
    plt.close(fig)

    print(f"  Pearson r(size, entropy) = {corr:+.4f}")


if __name__ == "__main__":
    main()
