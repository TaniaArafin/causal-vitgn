"""Evaluate a trained Causal-VITGN checkpoint.

Usage:
    PYTHONPATH=. python scripts/evaluate.py \
        --config config/default.yaml \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import CascadeDataset, collate_cascades
from src.models.causal_vitgn import CausalVITGN
from src.evaluation.metrics import all_metrics


def auto_configure_num_nodes(config: dict) -> dict:
    """Read num_users from metadata.json to keep model.num_nodes consistent."""
    metadata_path = os.path.join(config["data"]["processed_dir"], "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        num_users = metadata.get("num_users")
        if num_users is not None:
            config["model"]["num_nodes"] = int(num_users)
    return config


def main(config_path: str, checkpoint: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = auto_configure_num_nodes(config)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = CausalVITGN(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_path = os.path.join(config["data"]["processed_dir"], "test.pkl")
    test_ds = CascadeDataset(test_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_cascades,
        num_workers=0,
    )

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            out = model(batch)
            all_scores.append(out["intensity"].cpu().numpy())
            all_labels.append(batch["activated"].cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)
    # Decoder now returns raw logits; sigmoid -> probability.
    probs = 1.0 / (1.0 + np.exp(-scores))

    metrics = all_metrics(scores, labels, probs)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:8s} = {v:.4f}")

    # Print causal layer summary
    summary = model.causal_layer.adjacency_summary()
    print("\n=== Causal Adjacency Summary ===")
    for k, v in summary.items():
        print(f"  {k:18s} = {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = p.parse_args()
    main(args.config, args.checkpoint)
