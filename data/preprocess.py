"""Preprocess Higgs Twitter cascade data into model-ready tuples.

Run:
    python data/preprocess.py --raw_dir ./data/raw --out_dir ./data/processed
"""

import argparse
import gzip
import os
import pickle
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd


def load_higgs_retweets(path: str) -> pd.DataFrame:
    """Load retweet edges: src, dst, time."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    return pd.DataFrame(rows, columns=["src", "dst", "time"])


def reconstruct_cascades(
    retweets: pd.DataFrame,
    min_size: int = 5,
    max_size: int = 5000,
) -> List[Dict]:
    """Group retweets by destination user (cascade origin); each cascade is a
    chronological list of (source_user, time)."""
    cascades = defaultdict(list)
    rt_sorted = retweets.sort_values("time")
    for _, row in rt_sorted.iterrows():
        cascades[int(row["dst"])].append((int(row["src"]), int(row["time"])))

    filtered = []
    for origin, events in cascades.items():
        if min_size <= len(events) <= max_size:
            filtered.append({"origin": origin, "events": events})
    return filtered


def chronological_split(
    cascades: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    cs = sorted(cascades, key=lambda c: c["events"][0][1])
    n = len(cs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return cs[:train_end], cs[train_end:val_end], cs[val_end:]


def build_training_tuples(
    cascades: List[Dict],
    obs_window: float = 0.5,
    num_neighbors: int = 20,
    embed_dim: int = 128,
    latent_dim: int = 64,
) -> List[Dict]:
    """Convert cascades to model-ready tuples used by CascadeDataset.

    Each tuple represents a (target_user, current_time) prediction problem,
    drawn from the future portion of a cascade given the observed prefix.
    """
    tuples: List[Dict] = []
    for cas in cascades:
        events = cas["events"]
        n = len(events)
        split = max(1, int(n * obs_window))
        observed = events[:split]
        future = events[split:]

        if not observed or not future:
            continue

        for target_user, t_target in future:
            # Take last `num_neighbors` observed events as temporal neighbors
            recent = observed[-num_neighbors:]
            users = [e[0] for e in recent]
            times = [e[1] for e in recent]

            # Pad to fixed length
            while len(users) < num_neighbors:
                users.append(0)
                times.append(0.0)

            tuples.append({
                "target_node": target_user,
                "neighbors": users,
                "neighbor_times": times,
                "current_time": float(t_target),
                # Placeholders — filled at training time by the model
                "activated_embs": np.zeros((num_neighbors, embed_dim), dtype=np.float32),
                "z_neighbors": np.zeros((num_neighbors, latent_dim), dtype=np.float32),
                "time_since_last": float(t_target - times[-1]) if times[-1] > 0 else 0.0,
                "activated": 1.0,
                "interval": float(t_target - times[0]) if times[0] > 0 else 0.0,
            })
    return tuples


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(raw_dir: str, out_dir: str, obs_window: float):
    os.makedirs(out_dir, exist_ok=True)

    rt_path = os.path.join(raw_dir, "higgs-retweet_network.edgelist.gz")
    if not os.path.exists(rt_path):
        raise FileNotFoundError(
            f"{rt_path} not found. Run `python data/download.py` first."
        )

    print("Loading retweet network ...")
    rt = load_higgs_retweets(rt_path)
    print(f"  loaded {len(rt):,} retweet events")

    print("Reconstructing cascades ...")
    cascades = reconstruct_cascades(rt)
    print(f"  built {len(cascades):,} cascades after [5, 5000] filter")

    train, val, test = chronological_split(cascades)
    print(f"  split: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    for name, casc in [("train", train), ("val", val), ("test", test)]:
        tuples = build_training_tuples(casc, obs_window=obs_window)
        out_path = os.path.join(out_dir, f"{name}.pkl")
        save_pickle(tuples, out_path)
        print(f"  saved {len(tuples):,} {name} tuples -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="./data/raw")
    p.add_argument("--out_dir", default="./data/processed")
    p.add_argument("--obs_window", type=float, default=0.5)
    args = p.parse_args()
    main(args.raw_dir, args.out_dir, args.obs_window)
