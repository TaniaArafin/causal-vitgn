"""Fig. 6 — Risk-coverage curve.

Sort predictions by confidence (|p - 0.5|), then plot:
  x = coverage fraction (how many predictions we KEEP)
  y = error rate on the kept predictions

A useful selective-prediction property: the curve should be monotonically
increasing — the most confident predictions are more accurate, so the
error rate rises as we include less-confident ones.

Run:
    PYTHONPATH=. python scripts/figures/fig6_risk_coverage.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.figures._utils import load_model_and_test, save, setup_matplotlib
from src.utils.data_loader import collate_cascades


def collect_predictions(model, test_ds, device):
    loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
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


def main():
    setup_matplotlib()
    model, _, test_ds, device = load_model_and_test()
    probs, labels = collect_predictions(model, test_ds, device)

    # Confidence = distance from 0.5 (margin)
    confidence = np.abs(probs - 0.5)
    # Sort high-confidence first
    order = np.argsort(-confidence)
    probs_s = probs[order]
    labels_s = labels[order]

    preds_s = (probs_s >= 0.5).astype(int)
    correct_cum = np.cumsum(preds_s == labels_s)
    total_cum = np.arange(1, len(probs_s) + 1)
    acc_cum = correct_cum / total_cum
    error_cum = 1.0 - acc_cum
    coverage = total_cum / len(probs_s)

    # AUC of the risk-coverage curve (lower is better)
    auc = float(np.trapezoid(error_cum, coverage))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(coverage, error_cum, color="#2c7fb8", linewidth=2,
            label="Causal-VITGN")
    ax.axhline(error_cum[-1], color="#888", linestyle="--",
               label=f"Full-coverage error = {error_cum[-1]:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(error_cum.max(), 0.5) * 1.05)
    ax.set_xlabel("Coverage (fraction of kept predictions)")
    ax.set_ylabel("Error rate on kept predictions")
    ax.set_title("Risk–coverage curve · confidence = |p − 0.5|")
    ax.legend(loc="lower right", framealpha=0.95)

    ax.text(
        0.03, 0.95,
        f"AURC = {auc:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.35"),
    )

    save(fig, "fig6_risk_coverage")
    plt.close(fig)

    print(f"  AURC = {auc:.4f}")
    print(f"  Error @ 100% coverage: {error_cum[-1]:.4f}")
    if len(error_cum) >= 2:
        idx_50 = int(0.5 * len(error_cum))
        print(f"  Error @  50% coverage: {error_cum[idx_50]:.4f}")


if __name__ == "__main__":
    main()
