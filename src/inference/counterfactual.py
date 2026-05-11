"""Counterfactual inference engine for cascade prediction.

Implements Pearl's three-step procedure (abduction-action-prediction)
with Monte-Carlo sampling for uncertainty quantification.

Public API:
    CounterfactualEngine(model, num_samples=100)
        .remove_user(batch, user_id)
        .remove_users(batch, user_ids)
        .find_optimal_removal(batch, candidate_users, budget)

The engine is decoupled from training and operates on a pre-trained model.
"""

from typing import Dict, List, Sequence

import torch


class CounterfactualEngine:
    """Generates counterfactual cascade predictions via abduction-action-prediction."""

    def __init__(self, model, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples

    # ------------------------------------------------------------------ #
    #  Single-user removal
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def remove_user(
        self,
        batch: Dict[str, torch.Tensor],
        user_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Counterfactual: 'what if user_id had not reshared?'"""
        return self.remove_users(batch, [user_id])

    # ------------------------------------------------------------------ #
    #  Multi-user removal
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def remove_users(
        self,
        batch: Dict[str, torch.Tensor],
        user_ids: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """Counterfactual: 'what if these users had not reshared?'

        Sets their latent influence values to zero, severs incoming causal
        edges, and re-runs the Hawkes decoder. Repeats `num_samples` times
        to estimate uncertainty.
        """
        device = next(self.model.parameters()).device

        # Map user_ids -> latent component indices.
        # We use a deterministic hash into the latent space so the same user
        # always maps to the same component during evaluation.
        latent_dim = self.model.config["model"]["latent_dim"]
        latent_indices = [int(u) % latent_dim for u in user_ids]
        intervention_idx = torch.tensor(latent_indices, device=device, dtype=torch.long)
        intervention_val = torch.zeros(len(user_ids), device=device)

        cf_intensities = []
        for _ in range(self.num_samples):
            cf_int, _z_cf = self.model.counterfactual(
                batch, intervention_idx, intervention_val
            )
            cf_intensities.append(cf_int)

        stack = torch.stack(cf_intensities, dim=0)  # (S, B)
        return {
            "mean_intensity": stack.mean(dim=0),
            "std_intensity": stack.std(dim=0),
            "samples": stack,
        }

    # ------------------------------------------------------------------ #
    #  Greedy intervention search
    # ------------------------------------------------------------------ #
    def find_optimal_removal(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_users: List[int],
        budget: int,
    ) -> List[int]:
        """Greedy: select the k users whose removal minimizes predicted spread.

        Returns the ordered list of selected user_ids.
        """
        selected: List[int] = []
        remaining = list(candidate_users)

        for _ in range(budget):
            best_score = float("inf")
            best_user = None
            for u in remaining:
                trial = selected + [u]
                result = self.remove_users(batch, trial)
                score = float(result["mean_intensity"].sum())
                if score < best_score:
                    best_score = score
                    best_user = u
            if best_user is None:
                break
            selected.append(best_user)
            remaining.remove(best_user)

        return selected

    # ------------------------------------------------------------------ #
    #  Identity round-trip (used as a sanity test)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def identity_check(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Intervene with no users -> output should match the baseline.

        Returns the absolute difference between baseline and CF intensities.
        Should be close to zero up to MC sampling noise.
        """
        device = next(self.model.parameters()).device

        baseline = self.model(batch)
        baseline_intensity = baseline["intensity"]

        empty_idx = torch.zeros(0, dtype=torch.long, device=device)
        empty_val = torch.zeros(0, device=device)
        cf_int, _ = self.model.counterfactual(batch, empty_idx, empty_val)
        return (baseline_intensity - cf_int).abs()
