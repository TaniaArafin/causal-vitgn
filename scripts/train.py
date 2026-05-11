"""Train Causal-VITGN.

Usage:
    PYTHONPATH=. python scripts/train.py --config config/default.yaml
"""

import argparse
import os
import sys

import yaml
from torch.utils.data import DataLoader

# Make `src` importable when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import CascadeDataset, collate_cascades
from src.training.trainer import Trainer


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    proc_dir = config["data"]["processed_dir"]
    train_path = os.path.join(proc_dir, "train.pkl")
    val_path = os.path.join(proc_dir, "val.pkl")

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(
            f"Processed data not found in {proc_dir}. "
            "Run `python data/preprocess.py` first."
        )

    train_ds = CascadeDataset(train_path)
    val_ds = CascadeDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_cascades,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_cascades,
        num_workers=0,
    )

    trainer = Trainer(config, train_loader, val_loader)
    trainer.fit()
    print("\nTraining complete.")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoint saved to: {config['logging']['save_dir']}/best_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    args = p.parse_args()
    main(args.config)
