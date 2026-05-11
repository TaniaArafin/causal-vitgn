"""Neural Hawkes process decoder.

Computes intensity lambda_u(t) for each candidate user as a function of:
  - latent influence state z_u
  - time since last event for u
  - aggregated influence signal from already-activated neighbors (with decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralHawkesDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        time_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.latent_dim = latent_dim

        # Learnable exponential decay rate
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

        in_dim = latent_dim + time_dim + latent_dim  # z_u  +  Phi(dt)  +  neighbor_signal
        self.intensity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def time_encoding(self, dt: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time features for time-since-last."""
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        device = dt.device
        freqs = torch.linspace(0.0, 5.0, self.time_dim, device=device)
        return torch.sin(dt * freqs.unsqueeze(0))

    def neighbor_signal(
        self,
        z_neighbors: torch.Tensor,
        neighbor_times: torch.Tensor,
        current_time: torch.Tensor,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Exponentially-decayed aggregation of activated neighbor latents.

        Args:
            z_neighbors:    (B, K, latent_dim)
            neighbor_times: (B, K)
            current_time:   (B,)
            neighbor_mask:  (B, K) — True where padded
        """
        decay = F.softplus(self.log_decay)
        dt = (current_time.unsqueeze(-1) - neighbor_times).clamp(min=0)
        weights = torch.exp(-decay * dt)
        if neighbor_mask is not None:
            weights = weights.masked_fill(neighbor_mask, 0.0)
        weighted = z_neighbors * weights.unsqueeze(-1)
        return weighted.sum(dim=1)

    def forward(
        self,
        z_user: torch.Tensor,
        time_since_last: torch.Tensor,
        z_neighbors: torch.Tensor,
        neighbor_times: torch.Tensor,
        current_time: torch.Tensor,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns lambda: (B,) intensity values."""
        t_enc = self.time_encoding(time_since_last)
        n_signal = self.neighbor_signal(
            z_neighbors, neighbor_times, current_time, neighbor_mask
        )
        x = torch.cat([z_user, t_enc, n_signal], dim=-1)
        log_intensity = self.intensity_net(x).squeeze(-1)
        return F.softplus(log_intensity)

    def survival_probability(
        self, intensity: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """P(no event in dt) under piecewise-constant approximation."""
        return torch.exp(-intensity * dt)
