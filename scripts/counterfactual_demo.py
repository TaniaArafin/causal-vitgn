"""CLI counterfactual demo: print baseline + counterfactual predictions.

Usage:
    PYTHONPATH=. python scripts/counterfactual_demo.py \
        --config config/default.yaml \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import CascadeDataset, collate_cascades
from src.models.causal_vitgn import CausalVITGN
from src.inference.counterfactual import CounterfactualEngine


def main(config_path: str, checkpoint: str, num_cases: int = 3):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = CausalVITGN(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cf_engine = CounterfactualEngine(
        model, num_samples=config["counterfactual"]["num_samples"]
    )

    test_path = os.path.join(config["data"]["processed_dir"], "test.pkl")
    test_ds = CascadeDataset(test_path)
    loader = DataLoader(
        test_ds, batch_size=1, shuffle=True, collate_fn=collate_cascades
    )

    print("\n=== Counterfactual Demo ===\n")
    for i, batch in enumerate(loader):
        if i >= num_cases:
            break
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            baseline = model(batch)
        baseline_intensity = float(baseline["intensity"].sum())

        # Intervene on the first latent component (z_0 -> 0)
        idx = torch.tensor([0], device=device, dtype=torch.long)
        val = torch.zeros(1, device=device)
        cf_int, _ = model.counterfactual(batch, idx, val)
        cf_intensity = float(cf_int.sum())

        reduction = (baseline_intensity - cf_intensity) / max(baseline_intensity, 1e-6) * 100

        print(f"Case {i + 1}")
        print(f"  Baseline intensity:        {baseline_intensity:.4f}")
        print(f"  Counterfactual intensity:  {cf_intensity:.4f}")
        print(f"  Predicted reduction:       {reduction:+.1f}%\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--num_cases", type=int, default=3)
    args = p.parse_args()
    main(args.config, args.checkpoint, args.num_cases)
