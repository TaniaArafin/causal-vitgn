"""Fig. 3 — Calibration / reliability diagram.

Shows whether the model's predicted probabilities match the actual
observed activation rate. Perfectly calibrated = diagonal line.

Run:
    PYTHONPATH=. python scripts/figures/fig3_calibration.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.figures._utils import load_model_and_test, save, setup_matplotlib
from src.utils.data_loader import collate_cascades


def collect_predictions(model, test_ds, device, batch_size: int = 128):
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_cascades,
    )
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model(batch)
            p = torch.sigmoid(out["intensity"]).cpu().numpy()
            y = batch["activated"].cpu().numpy()
            all_p.append(p)
            all_y.append(y)
    return np.concatenate(all_p), np.concatenate(all_y).astype(int)


def reliability_bins(probs, labels, num_bins: int = 10):
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers, bin_acc, bin_conf, bin_count = [], [], [], []
    for i in range(num_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == num_bins - 1:
            mask |= probs == 1.0
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        bin_acc.append(labels[mask].mean())
        bin_conf.append(probs[mask].mean())
        bin_count.append(int(mask.sum()))
    return (
        np.array(centers),
        np.array(bin_acc),
        np.array(bin_conf),
        np.array(bin_count),
    )


def ece(probs, labels, num_bins: int = 10):
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = len(probs)
    total = 0.0
    for i in range(num_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(
            labels[mask].mean() - probs[mask].mean()
        )
    return total


def main():
    setup_matplotlib()
    model, _, test_ds, device = load_model_and_test()
    probs, labels = collect_predictions(model, test_ds, device)
    centers, bin_acc, bin_conf, bin_count = reliability_bins(probs, labels)
    ece_val = ece(probs, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # ----- Left: reliability diagram -----
    ax1.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    ax1.bar(centers, bin_acc, width=0.08, alpha=0.7,
            color="#2c7fb8", edgecolor="#1a4f76",
            label="observed frequency")
    ax1.plot(bin_conf, bin_acc, "o-", color="#d62728",
             markersize=7, linewidth=1.4,
             label="model")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Empirical activation rate")
    ax1.set_title(f"Reliability diagram (ECE = {ece_val:.3f})")
    ax1.legend(loc="upper left", framealpha=0.9)

    # ----- Right: confidence histogram -----
    ax2.bar(centers, bin_count, width=0.08, color="#5b9bd5",
            edgecolor="#1a4f76", alpha=0.85)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("# test samples")
    ax2.set_title("Confidence distribution")

    fig.suptitle("Calibration on Higgs Twitter test set", fontsize=12, y=1.02)
    save(fig, "fig3_calibration")
    plt.close(fig)

    print(f"  ECE = {ece_val:.4f}")


if __name__ == "__main__":
    main()
