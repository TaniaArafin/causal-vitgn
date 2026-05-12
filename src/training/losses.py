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




def cascade_nll(
   intensities: torch.Tensor,
   activated: torch.Tensor,
   interval: torch.Tensor,
   eps: float = 1e-6,
) -> torch.Tensor:
   """Negative log-likelihood for cascade prediction with negative sampling.


   Treats intensity as the rate of a Poisson event in an interval of length
   `interval`. The probability of activation is:


       p_activated = 1 - exp(-lambda * interval)


   so the BCE loss is:


       -[ y * log(p) + (1-y) * log(1-p) ]
     = -[ y * log(1 - exp(-lambda*interval)) - (1-y) * lambda * interval ]


   This is well-defined for both positive (y=1) and negative (y=0) samples,
   unlike the raw Hawkes log-likelihood which assumes every example is an
   observed event.


   Stability:
     * intensities clamped to [eps, 1e4]
     * intervals clamped to [0, 1e4]
     * lam*dt clamped to [eps, 30] so exp(-lam*dt) stays in float range
     * Any NaN/Inf in the per-sample log-lik is replaced with 0 and the
       offending sample contributes a constant penalty, so a single bad
       example cannot poison the whole batch.
   """
   # Hard clamps on inputs
   intensities = intensities.clamp(min=eps, max=1e4)
   interval = interval.clamp(min=0.0, max=1e4)


   # lam * dt in a range where exp() is finite
   lam_dt = (intensities * interval).clamp(min=eps, max=30.0)


   # log p(activated=1) = log(1 - exp(-lambda*dt))   numerically stable
   log_p_pos = torch.log1p(-torch.exp(-lam_dt) + eps)


   # log p(activated=0) = -lambda * dt
   log_p_neg = -lam_dt


   log_lik = activated * log_p_pos + (1.0 - activated) * log_p_neg


   # Replace any residual nan/inf with a finite penalty so training survives.
   log_lik = torch.nan_to_num(log_lik, nan=-10.0, posinf=0.0, neginf=-10.0)


   return -log_lik.mean()




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





