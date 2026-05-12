"""Preprocess Higgs Twitter cascade data into model-ready tuples.


Run:
   python data/preprocess.py --raw_dir ./data/raw --out_dir ./data/processed


Key features:
 * Reconstructs cascades from raw retweet edges.
 * Remaps user IDs to a compact range [0, num_users) so the model's
   nn.Embedding can index them without going out of bounds.
 * Writes metadata.json with the final num_users (so training code can
   auto-configure model.num_nodes).
"""


import argparse
import gzip
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple


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




def build_user_remap(retweets: pd.DataFrame) -> Tuple[Dict[int, int], int]:
   """Map every original user id to a compact id in [0, num_users).


   Reserves id 0 as a PAD token, so real users start at 1.
   """
   all_users = set(retweets["src"].unique()) | set(retweets["dst"].unique())
   sorted_users = sorted(all_users)


   remap: Dict[int, int] = {0: 0}  # PAD id stays at 0
   for i, u in enumerate(sorted_users, start=1):
       remap[u] = i


   # num_users = number of unique users + 1 PAD slot
   num_users = len(sorted_users) + 1
   return remap, num_users




def remap_retweets(retweets: pd.DataFrame, remap: Dict[int, int]) -> pd.DataFrame:
   """Apply id remapping to a retweets DataFrame."""
   rt = retweets.copy()
   rt["src"] = rt["src"].map(remap)
   rt["dst"] = rt["dst"].map(remap)
   return rt




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
   num_users: int,
   obs_window: float = 0.5,
   num_neighbors: int = 20,
   embed_dim: int = 128,
   latent_dim: int = 64,
   neg_ratio: int = 1,
   rng_seed: int = 42,
) -> List[Dict]:
   """Convert cascades to model-ready tuples used by CascadeDataset.


   Each cascade contributes (target_user, current_time) examples drawn from
   its future portion. For every positive example we also emit `neg_ratio`
   negative examples — random users who did NOT retweet in this cascade —
   so the classifier has both classes to learn from.


   All user ids are GUARANTEED to be in [0, num_users).
   """
   rng = np.random.default_rng(rng_seed)
   tuples: List[Dict] = []


   for cas in cascades:
       events = cas["events"]
       n = len(events)
       split = max(1, int(n * obs_window))
       observed = events[:split]
       future = events[split:]


       if not observed or not future:
           continue


       cascade_users = {e[0] for e in events}


       for target_user, t_target in future:
           if not (0 <= target_user < num_users):
               continue


           recent = observed[-num_neighbors:]
           users = [e[0] for e in recent if 0 <= e[0] < num_users]
           times = [e[1] for e in recent if 0 <= e[0] < num_users]


           if not users:
               continue


           while len(users) < num_neighbors:
               users.append(0)
               times.append(0.0)
           users = users[:num_neighbors]
           times = times[:num_neighbors]


           base = {
               "neighbors": [int(u) for u in users],
               "neighbor_times": [float(t) for t in times],
               "current_time": float(t_target),
               "activated_embs": np.zeros((num_neighbors, embed_dim), dtype=np.float32),
               "z_neighbors": np.zeros((num_neighbors, latent_dim), dtype=np.float32),
               "time_since_last": float(t_target - times[-1]) if times[-1] > 0 else 0.0,
               "interval": float(t_target - times[0]) if times[0] > 0 else 0.0,
           }


           tuples.append({
               **base,
               "target_node": int(target_user),
               "activated": 1.0,
           })


           for _ in range(neg_ratio):
               for _ in range(10):
                   neg = int(rng.integers(1, num_users))
                   if neg not in cascade_users:
                       break
               else:
                   continue
               tuples.append({
                   **base,
                   "target_node": neg,
                   "activated": 0.0,
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


   print("Building user-id remap ...")
   remap, num_users = build_user_remap(rt)
   print(f"  {num_users:,} unique users (including PAD id 0)")
   rt = remap_retweets(rt, remap)


   print("Reconstructing cascades ...")
   cascades = reconstruct_cascades(rt)
   print(f"  built {len(cascades):,} cascades after [5, 5000] filter")


   train, val, test = chronological_split(cascades)
   print(f"  split: train={len(train):,} | val={len(val):,} | test={len(test):,}")


   for name, casc in [("train", train), ("val", val), ("test", test)]:
       tuples = build_training_tuples(casc, num_users=num_users, obs_window=obs_window)
       out_path = os.path.join(out_dir, f"{name}.pkl")
       save_pickle(tuples, out_path)
       print(f"  saved {len(tuples):,} {name} tuples -> {out_path}")


   # Write metadata so training code can auto-configure model.num_nodes
   metadata = {
       "num_users": num_users,
       "obs_window": obs_window,
       "cascade_count": {
           "train": len(train),
           "val": len(val),
           "test": len(test),
       },
   }
   with open(os.path.join(out_dir, "metadata.json"), "w") as f:
       json.dump(metadata, f, indent=2)
   print(f"\n  wrote metadata.json with num_users={num_users:,}")
   print(f"  -> Update config/default.yaml: model.num_nodes = {num_users}")




if __name__ == "__main__":
   p = argparse.ArgumentParser()
   p.add_argument("--raw_dir", default="./data/raw")
   p.add_argument("--out_dir", default="./data/processed")
   p.add_argument("--obs_window", type=float, default=0.5)
   args = p.parse_args()
   main(args.raw_dir, args.out_dir, args.obs_window)


