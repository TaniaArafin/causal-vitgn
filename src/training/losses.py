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
import torch.nn.functional as F




def cascade_nll(
   logits: torch.Tensor,
   activated: torch.Tensor,
   interval: torch.Tensor = None,  # unused, kept for API compatibility
   eps: float = 1e-6,
) -> torch.Tensor:
   """Binary cross-entropy on raw logits for cascade activation.


   The decoder returns an unbounded logit l. We model
       P(activated=1) = sigmoid(l)
   and minimise the standard BCE loss. This uses PyTorch's fused
   BCEWithLogitsLoss which is numerically stable for any finite l and,
   crucially, has full gradient signal everywhere — no clamps, no
   saturation, nothing that can stall training the way the previous
   Poisson-survival formulation did.


   interval is accepted for backwards compatibility but ignored; if you
   want a Hawkes-style intensity for the paper, compute it post-hoc as
   softplus(logit) / interval.
   """
   return F.binary_cross_entropy_with_logits(logits, activated, reduction="mean")




def hawkes_log_likelihood(
   intensities: torch.Tensor,
   activated: torch.Tensor,
   interval: torch.Tensor,
   eps: float = 1e-8,
) -> torch.Tensor:
   """Kept for backwards compatibility. Use cascade_nll for new code."""
   return -cascade_nll(intensities, activated, interval, eps)




def kl_gaussian(
   mu: torch.Tensor,
   logvar: torch.Tensor,
   free_bits: float = 0.0,
) -> torch.Tensor:
   """KL[ N(mu, sigma^2) || N(0, I) ].


   Computes per-dimension KL, applies a free-bits floor (every latent
   dimension must carry at least `free_bits` nats), then averages over
   the batch and sums over latent dim. Free-bits prevents posterior
   collapse: when KL per dim is below the floor, no gradient flows through
   that dim's prior, so the model is allowed to use information freely.
   """
   logvar = logvar.clamp(min=-10.0, max=10.0)
   # Per-dim KL, shape (B, D)
   kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
   # Free bits: clamp per-dim KL from below so the optimiser stops driving
   # tiny KLs toward zero.
   if free_bits > 0.0:
       kl_per_dim = kl_per_dim.clamp(min=free_bits)
   return kl_per_dim.sum(dim=-1).mean()




class CausalVITGNLoss(nn.Module):
   """Combined ELBO + DAG + sparsity loss with free-bits KL and warmup."""


   def __init__(
       self,
       dag_penalty: float = 1.0,
       sparsity_penalty: float = 0.01,
       consistency_penalty: float = 0.5,
       free_bits: float = 0.0,
   ):
       super().__init__()
       self.dag_penalty = dag_penalty
       self.sparsity_penalty = sparsity_penalty
       self.consistency_penalty = consistency_penalty
       self.free_bits = free_bits


   def forward(
       self,
       outputs: Dict[str, torch.Tensor],
       batch: Dict[str, torch.Tensor],
       causal_layer,
       beta: float = 1.0,
       structural_weight: float = 1.0,
   ) -> Dict[str, torch.Tensor]:
       """structural_weight (0..1) ramps the DAG + sparsity penalties.
       Letting NLL settle before structural constraints kick in is what
       prevents the optimiser from collapsing A to zero on epoch 0.
       """
       nll = cascade_nll(
           outputs["intensity"],
           batch["activated"],
           batch["interval"],
       )


       kl = kl_gaussian(outputs["mu"], outputs["logvar"], free_bits=self.free_bits)


       h_A = causal_layer.get_dag_constraint()
       dag_loss = h_A * h_A
       sparsity_loss = causal_layer.get_sparsity_penalty()


       total = (
           nll
           + beta * kl
           + structural_weight * self.dag_penalty * dag_loss
           + structural_weight * self.sparsity_penalty * sparsity_loss
       )


       return {
           "total": total,
           "nll": nll.detach(),
           "kl": kl.detach(),
           "dag": dag_loss.detach(),
           "sparsity": sparsity_loss.detach(),
           "h_A": h_A.detach(),
       }





