"""Loss functions for Causal-VITGN.

Total loss:
    L = L_NLL                       (Hawkes log-likelihood)
      + beta * KL[q(z|h,c) || p(z)]
      + lambda_DAG * h(A)^2          (NOTEARS acyclicity)
      + lambda_sparse * ||A||_1      (sparse causal graph)
"""

from typing import Dict

import torch
import torch.nn as nn


def hawkes_log_likelihood(
    intensities: torch.Tensor,
    activated: torch.Tensor,
    interval: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-likelihood of activation events under a piecewise-constant intensity.

    For each event:
        log p = activated * log(lambda) - lambda * interval

    Returns:
        Total log-likelihood (sum over batch).
    """
    log_event = activated * torch.log(intensities + eps)
    survival = -intensities * interval
    return (log_event + survival).sum()


def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL[ N(mu, sigma^2) || N(0, I) ] summed over batch and latent dim."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class CausalVITGNLoss(nn.Module):
    """Combined ELBO + DAG + sparsity loss."""

    def __init__(
        self,
        dag_penalty: float = 1.0,
        sparsity_penalty: float = 0.01,
        consistency_penalty: float = 0.5,
    ):
        super().__init__()
        self.dag_penalty = dag_penalty
        self.sparsity_penalty = sparsity_penalty
        self.consistency_penalty = consistency_penalty

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        causal_layer,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        # 1) Negative Hawkes log-likelihood
        nll = -hawkes_log_likelihood(
            outputs["intensity"],
            batch["activated"],
            batch["interval"],
        )

        # 2) KL divergence
        kl = kl_gaussian(outputs["mu"], outputs["logvar"])

        # 3) DAG acyclicity penalty (squared so it stays smooth)
        h_A = causal_layer.get_dag_constraint()
        dag_loss = h_A * h_A

        # 4) Sparsity penalty
        sparsity_loss = causal_layer.get_sparsity_penalty()

        total = (
            nll
            + beta * kl
            + self.dag_penalty * dag_loss
            + self.sparsity_penalty * sparsity_loss
        )

        return {
            "total": total,
            "nll": nll.detach(),
            "kl": kl.detach(),
            "dag": dag_loss.detach(),
            "sparsity": sparsity_loss.detach(),
            "h_A": h_A.detach(),
        }
