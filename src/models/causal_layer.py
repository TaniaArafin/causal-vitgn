"""Structural Causal Layer (Pearl's framework) for counterfactual reasoning.

This module implements a linear Structural Causal Model (SCM) over a batch of
latent influence states z_i:

    z = A * z + epsilon         (structural equations)
    <=> z = (I - A)^{-1} * epsilon  (when A is acyclic)

The adjacency A is constrained to be a directed acyclic graph (DAG) via the
NOTEARS continuous optimization criterion [Zheng et al., 2018]:

    h(A) = trace(exp(A o A)) - N    (h(A) = 0 iff A is acyclic)

Pearl's three-step counterfactual procedure (abduction-action-prediction) is
exposed via the `intervene()` method for use by CounterfactualEngine.

References
----------
Pearl, J. (2009). Causality. Cambridge Univ. Press.
Zheng et al. (2018). DAGs with NO TEARS. NeurIPS.
"""

from typing import Optional

import torch
import torch.nn as nn


class StructuralCausalLayer(nn.Module):
    """Linear SCM with a DAG-constrained learnable adjacency matrix.

    The layer operates on a batch of (n, d) latent vectors, where n is the
    number of latent units (in our case, the latent influence dimension)
    and d is feature dimension. We treat the n axis as the causal variables
    that A connects.

    Notes
    -----
    For very large n, computing matrix_exp on (n, n) is expensive. For typical
    latent_dim values (32-128), this is fine.
    """

    def __init__(
        self,
        latent_dim: int,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Initialize A close to zero so the SCM starts as identity-like
        A0 = init_scale * torch.randn(latent_dim, latent_dim)
        A0.fill_diagonal_(0.0)
        self.A = nn.Parameter(A0)

    # ------------------------------------------------------------------ #
    #  Constraint terms
    # ------------------------------------------------------------------ #
    def get_dag_constraint(self) -> torch.Tensor:
        """NOTEARS acyclicity penalty h(A).

        h(A) = trace(exp(A o A)) - N
        h(A) = 0  <=>  A is a DAG.
        """
        A_squared = self.A * self.A
        # matrix_exp expects a square matrix; this is differentiable.
        h = torch.trace(torch.matrix_exp(A_squared)) - self.latent_dim
        return h

    def get_sparsity_penalty(self) -> torch.Tensor:
        """L1 norm of A entries — encourages a sparse causal graph."""
        return self.A.abs().sum()

    # ------------------------------------------------------------------ #
    #  Forward pass: structural equation
    # ------------------------------------------------------------------ #
    def _solve(self, IA: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Solve (I - A) z = rhs for z.

        Args:
            IA:  (n, n)
            rhs: (B, n) or (B, n, d)
        """
        if rhs.dim() == 2:
            return torch.linalg.solve(IA, rhs.t()).t()
        elif rhs.dim() == 3:
            B, n, d = rhs.shape
            flat = rhs.permute(1, 0, 2).reshape(n, B * d)  # (n, B*d)
            sol = torch.linalg.solve(IA, flat)             # (n, B*d)
            return sol.reshape(n, B, d).permute(1, 0, 2)
        else:
            raise ValueError(f"Unsupported rhs shape: {rhs.shape}")

    def forward(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Run the structural equations: z = (I - A)^{-1} * epsilon.

        Args:
            epsilon: (B, n) or (B, n, d) exogenous noise
        Returns:
            z with the same shape
        """
        I = torch.eye(self.latent_dim, device=self.A.device, dtype=self.A.dtype)
        IA = I - self.A
        return self._solve(IA, epsilon)

    # ------------------------------------------------------------------ #
    #  Pearl: abduction
    # ------------------------------------------------------------------ #
    def abduce(self, z_observed: torch.Tensor) -> torch.Tensor:
        """Recover exogenous noise epsilon from an observation.

        epsilon = (I - A) @ z_observed

        Args:
            z_observed: (B, n) or (B, n, d)
        """
        I = torch.eye(self.latent_dim, device=self.A.device, dtype=self.A.dtype)
        IA = I - self.A
        if z_observed.dim() == 2:
            return torch.matmul(z_observed, IA.t())
        elif z_observed.dim() == 3:
            return torch.einsum("ij,bjd->bid", IA, z_observed)
        else:
            raise ValueError(f"Unsupported z shape: {z_observed.shape}")

    # ------------------------------------------------------------------ #
    #  Pearl: intervention (do-operator)
    # ------------------------------------------------------------------ #
    def _modified_adjacency(
        self, intervention_indices: torch.Tensor
    ) -> torch.Tensor:
        """Sever incoming edges to intervened nodes (do-operator)."""
        A_mod = self.A.clone()
        A_mod[intervention_indices, :] = 0.0
        return A_mod

    def intervene(
        self,
        z_observed: torch.Tensor,
        intervention_indices: torch.Tensor,
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """Pearl's three-step counterfactual procedure.

        Steps:
          1. Abduction:   epsilon* = (I - A) z_observed
          2. Action:      modify A to sever incoming edges of intervened nodes
          3. Prediction:  z^cf = (I - A_modified)^{-1} epsilon*
                          override z^cf at intervened indices with x'

        Args:
            z_observed: (n,) or (B, n) or (B, n, d)
            intervention_indices: (k,) integer indices into the n axis
            intervention_values:  scalar, (k,), (B, k), or (B, k, d)
        Returns:
            z_cf with same shape as z_observed
        """
        single = z_observed.dim() == 1
        if single:
            z_observed = z_observed.unsqueeze(0)  # (1, n) for uniform handling

        epsilon_star = self.abduce(z_observed)

        A_mod = self._modified_adjacency(intervention_indices)
        I = torch.eye(self.latent_dim, device=self.A.device, dtype=self.A.dtype)
        IA_mod = I - A_mod
        z_cf = self._solve(IA_mod, epsilon_star)

        # Force intervened values
        z_cf = self._force_values(z_cf, intervention_indices, intervention_values)

        if single:
            z_cf = z_cf.squeeze(0)
        return z_cf

    def _force_values(
        self,
        z: torch.Tensor,
        idx: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Override z[..., idx, ...] with the supplied intervention values.

        Accepted `values` shapes:
            * z is (B, n):    values may be (k,), (B, k), or scalar.
            * z is (B, n, d): values may be (k, d), (B, k, d), or scalar.
        """
        if idx.numel() == 0:
            return z

        if z.dim() == 2:
            B, _ = z.shape
            if values.dim() == 0:
                broadcast_val = values.expand(B, idx.numel())
            elif values.dim() == 1:
                broadcast_val = values.unsqueeze(0).expand(B, -1)
            else:
                broadcast_val = values
            z[:, idx] = broadcast_val
        elif z.dim() == 3:
            B, _, d = z.shape
            if values.dim() == 0:
                broadcast_val = values.expand(B, idx.numel(), d)
            elif values.dim() == 2:
                # (k, d) -> (B, k, d)
                broadcast_val = values.unsqueeze(0).expand(B, -1, -1)
            else:
                broadcast_val = values
            z[:, idx, :] = broadcast_val
        else:
            raise ValueError(f"Unsupported z shape: {z.shape}")
        return z

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def adjacency_summary(self) -> dict:
        """Quick summary of the learned causal graph."""
        A = self.A.detach().cpu()
        return {
            "n_nonzero": int((A.abs() > 1e-3).sum()),
            "max_abs": float(A.abs().max()),
            "frobenius_norm": float(A.norm()),
            "dag_constraint": float(self.get_dag_constraint().detach()),
        }
