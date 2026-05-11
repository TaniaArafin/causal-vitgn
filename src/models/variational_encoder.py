"""Variational Influence Encoder.

Produces a Gaussian posterior q(z | h, c) over latent influence states,
with reparameterization sampling and a closed-form KL term.
"""

import torch
import torch.nn as nn


class CascadeContextEncoder(nn.Module):
    """Attention-pooled summary of already-activated users in a cascade."""

    def __init__(self, embed_dim: int, context_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, context_dim)
        self.scale = embed_dim ** -0.5

    def forward(
        self,
        target_emb: torch.Tensor,
        activated_embs: torch.Tensor,
        activated_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            target_emb:     (B, embed_dim)
            activated_embs: (B, M, embed_dim)
            activated_mask: (B, M) — True where padded
        Returns:
            context: (B, context_dim)
        """
        Q = self.q_proj(target_emb).unsqueeze(1)              # (B, 1, D)
        K = self.k_proj(activated_embs)                       # (B, M, D)
        V = self.v_proj(activated_embs)                       # (B, M, C)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, 1, M)
        if activated_mask is not None:
            scores = scores.masked_fill(activated_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # Replace NaNs from all-padded rows with zeros
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, V).squeeze(1)


class VariationalEncoder(nn.Module):
    """q_phi(z | h, c) = N(mu, sigma^2)."""

    def __init__(
        self,
        embed_dim: int,
        context_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        in_dim = embed_dim + context_dim
        hidden = max(in_dim, latent_dim)
        self.mu_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
    ):
        x = torch.cat([h, context], dim=-1)
        mu = self.mu_net(x)
        logvar = self.logvar_net(x).clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL[ N(mu, sigma^2) || N(0, I) ] summed over latent dim."""
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )
