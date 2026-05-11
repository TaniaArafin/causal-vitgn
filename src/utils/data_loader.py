"""Cascade dataset loader."""

import pickle
from typing import List

import torch
from torch.utils.data import Dataset


class CascadeDataset(Dataset):
    """Loads preprocessed cascade tuples from a pickle file."""

    def __init__(self, processed_path: str):
        with open(processed_path, "rb") as f:
            self.cascades = pickle.load(f)

    def __len__(self) -> int:
        return len(self.cascades)

    def __getitem__(self, idx: int) -> dict:
        c = self.cascades[idx]
        return {
            "target_nodes": torch.tensor(c["target_node"], dtype=torch.long),
            "neighbor_nodes": torch.tensor(c["neighbors"], dtype=torch.long),
            "neighbor_times": torch.tensor(c["neighbor_times"], dtype=torch.float),
            "current_time": torch.tensor(c["current_time"], dtype=torch.float),
            "activated_embs": torch.tensor(c["activated_embs"], dtype=torch.float),
            "z_neighbors": torch.tensor(c["z_neighbors"], dtype=torch.float),
            "time_since_last": torch.tensor(c["time_since_last"], dtype=torch.float),
            "activated": torch.tensor(c["activated"], dtype=torch.float),
            "interval": torch.tensor(c["interval"], dtype=torch.float),
        }


def collate_cascades(batch: List[dict]) -> dict:
    """Collate a list of cascade samples into a batched dict."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        # Scalar tensors -> stack
        if vals[0].dim() == 0:
            out[k] = torch.stack(vals)
        else:
            # Pad along dim 0 if shapes differ
            shapes = {tuple(v.shape) for v in vals}
            if len(shapes) == 1:
                out[k] = torch.stack(vals)
            else:
                max_len = max(v.size(0) for v in vals)
                pad_shape = list(vals[0].shape)
                pad_shape[0] = max_len
                padded = torch.zeros(len(vals), *pad_shape, dtype=vals[0].dtype)
                for i, v in enumerate(vals):
                    padded[i, : v.size(0)] = v
                out[k] = padded
    return out
