"""Predictive evaluation metrics for cascade prediction."""

import numpy as np
from sklearn.metrics import roc_auc_score


def map_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean Average Precision @ k for a single query."""
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k = order[:k]
    hits = labels[top_k]
    if hits.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(hits)
    precision_at_i = cum_hits / np.arange(1, len(top_k) + 1)
    return float((precision_at_i * hits).sum() / hits.sum())


def hits_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 50) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k = order[:k]
    return float(labels[top_k].sum() / max(labels.sum(), 1))


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def mae_time(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(t_true - t_pred)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE) over `num_bins` equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(num_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray = None,
):
    if probs is None:
        # Convert intensity to probability via 1 - exp(-lambda)
        probs = 1.0 - np.exp(-scores)
        probs = np.clip(probs, 0.0, 1.0)

    return {
        "MAP@10": map_at_k(scores, labels, 10),
        "MAP@50": map_at_k(scores, labels, 50),
        "Hits@50": hits_at_k(scores, labels, 50),
        "AUROC": auroc(scores, labels),
        "ECE": expected_calibration_error(probs, labels),
    }
