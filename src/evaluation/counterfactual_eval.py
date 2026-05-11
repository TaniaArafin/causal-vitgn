"""Counterfactual evaluation protocol.

Three counterfactual metrics:
  1. CF MAE              -- counterfactual cascade size error
  2. Plausibility score  -- fraction of CF predictions that move in the
                            expected direction (e.g. removing a high-
                            influence user should reduce predicted spread)
  3. Intervention effectiveness -- percent reduction vs. baseline
"""

from typing import Dict, List

import numpy as np


def cf_size_mae(
    predicted_sizes: np.ndarray, true_sizes: np.ndarray
) -> float:
    """Mean absolute error between CF prediction and synthetic ground truth."""
    if len(predicted_sizes) == 0:
        return 0.0
    return float(np.mean(np.abs(predicted_sizes - true_sizes)))


def plausibility_score(
    cf_predictions: np.ndarray,
    observed_baseline: np.ndarray,
) -> float:
    """Fraction of CF predictions moving in the expected direction.

    For 'remove influencer' interventions, we expect cf < baseline.
    """
    if len(cf_predictions) == 0:
        return 0.0
    moved_correctly = (cf_predictions < observed_baseline).astype(float)
    return float(moved_correctly.mean())


def intervention_effectiveness(
    baseline_spread: float, cf_spread: float
) -> float:
    """Percent reduction in predicted spread (positive = effective)."""
    if baseline_spread <= 0:
        return 0.0
    return float((baseline_spread - cf_spread) / baseline_spread)


def evaluate_counterfactual_protocol(
    model,
    cf_engine,
    test_cases: List[Dict],
) -> Dict[str, float]:
    """Run a batch of counterfactual queries and aggregate metrics.

    Each test case is:
        {
            "batch": dict,
            "user_id": int,
            "ground_truth_size": float,
        }
    """
    pred_sizes, true_sizes, baselines = [], [], []

    for case in test_cases:
        batch = case["batch"]
        u = case["user_id"]
        gt = case["ground_truth_size"]

        baseline = model(batch)
        baseline_spread = float(baseline["intensity"].sum())

        cf_result = cf_engine.remove_user(batch, u)
        cf_spread = float(cf_result["mean_intensity"].sum())

        pred_sizes.append(cf_spread)
        true_sizes.append(gt)
        baselines.append(baseline_spread)

    pred = np.array(pred_sizes)
    true = np.array(true_sizes)
    base = np.array(baselines)

    return {
        "CF_MAE": cf_size_mae(pred, true),
        "Plausibility": plausibility_score(pred, base),
        "Avg_Intervention_Reduction_pct": float(
            np.mean(
                [
                    intervention_effectiveness(b, p)
                    for b, p in zip(base, pred)
                ]
            )
            * 100
        ),
    }
