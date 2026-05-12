"""Shared helpers for paper figure generation.

Designed to run identically in:
  - Google Colab (auto-detects /content/causal-vitgn)
  - Local Mac (uses the project root)

Public API:
  setup_matplotlib()           — apply IEEE-friendly plot defaults
  load_model_and_test(cfg, ckpt) -> (model, config, test_dataset, device)
  save(fig, name)              — write figures/<name>.pdf + .png
  FIGURES_DIR                  — absolute path to the figures/ folder
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import yaml


# --------------------------------------------------------------------- #
#  Project paths (works on Colab + local Mac)
# --------------------------------------------------------------------- #
def _detect_project_root() -> Path:
    candidates = [
        Path("/content/causal-vitgn"),                       # Colab
        Path(__file__).resolve().parent.parent.parent,        # repo-relative
    ]
    for c in candidates:
        if (c / "config" / "default.yaml").exists():
            return c
    raise RuntimeError("Could not locate the causal-vitgn project root.")


PROJECT_ROOT = _detect_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- #
#  Matplotlib styling
# --------------------------------------------------------------------- #
def setup_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


# --------------------------------------------------------------------- #
#  Model + data loading (cached at module level)
# --------------------------------------------------------------------- #
def load_model_and_test(
    config_path: Path = DEFAULT_CONFIG,
    ckpt_path: Path = DEFAULT_CKPT,
):
    from src.models.causal_vitgn import CausalVITGN
    from src.utils.data_loader import CascadeDataset

    with open(config_path) as f:
        config = yaml.safe_load(f)

    metadata_path = Path(config["data"]["processed_dir"]) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        config["model"]["num_nodes"] = int(meta["num_users"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = CausalVITGN(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = CascadeDataset(
        os.path.join(config["data"]["processed_dir"], "test.pkl")
    )
    return model, config, test_ds, device


# --------------------------------------------------------------------- #
#  Saving
# --------------------------------------------------------------------- #
def save(fig: plt.Figure, name: str):
    """Save a matplotlib figure as both PDF (for LaTeX) and PNG (for slides)."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"  saved {pdf_path.name} + {png_path.name}")
