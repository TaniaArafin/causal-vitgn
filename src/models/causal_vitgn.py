"""Causal-VITGN: full model assembly.

Pipeline:
    Temporal Graph Encoder
        -> Variational Influence Encoder
            -> [Structural Causal Layer]   <-- counterfactual extension
                -> Neural Hawkes Decoder

The Structural Causal Layer operates over the LATENT DIMENSION of z, treating
each latent component as a causal variable connected by the learned DAG A.
"""

from typing import Dict

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .variational_encoder import CascadeContextEncoder, VariationalEncoder
from .causal_layer import StructuralCausalLayer
from .hawkes_decoder import NeuralHawkesDecoder


class CausalVITGN(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        m = config["model"]
        num_nodes = m["num_nodes"]

        self.temporal_encoder = TemporalEncoder(
            num_nodes=num_nodes,
            memory_dim=m["memory_dim"],
            embed_dim=m["embedding_dim"],
            time_dim=m["time_encoding_dim"],
            num_heads=m["num_heads"],
            dropout=m["dropout"],
        )

        self.context_encoder = CascadeContextEncoder(
            embed_dim=m["embedding_dim"],
            context_dim=m["embedding_dim"],
        )

        self.variational_encoder = VariationalEncoder(
            embed_dim=m["embedding_dim"],
            context_dim=m["embedding_dim"],
            latent_dim=m["latent_dim"],
        )

        # Structural Causal Model layer over the latent dimension.
        self.causal_layer = StructuralCausalLayer(
            latent_dim=m["latent_dim"],
        )

        self.hawkes_decoder = NeuralHawkesDecoder(
            latent_dim=m["latent_dim"],
            time_dim=m["time_encoding_dim"],
        )

    # ------------------------------------------------------------------ #
    #  Encoding pipeline (associational)
    # ------------------------------------------------------------------ #
    def encode(self, batch: Dict[str, torch.Tensor]):
        h = self.temporal_encoder(
            target_nodes=batch["target_nodes"],
            neighbor_nodes=batch["neighbor_nodes"],
            neighbor_times=batch["neighbor_times"],
            current_time=batch["current_time"],
            neighbor_mask=batch.get("neighbor_mask"),
        )
        context = self.context_encoder(
            target_emb=h,
            activated_embs=batch["activated_embs"],
            activated_mask=batch.get("activated_mask"),
        )
        mu, logvar = self.variational_encoder(h, context)
        z = self.variational_encoder.reparameterize(mu, logvar)
        return h, mu, logvar, z

    def predict_intensity(
        self, z: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.hawkes_decoder(
            z_user=z,
            time_since_last=batch["time_since_last"],
            z_neighbors=batch["z_neighbors"],
            neighbor_times=batch["neighbor_times"],
            current_time=batch["current_time"],
            neighbor_mask=batch.get("neighbor_mask"),
        )

    # ------------------------------------------------------------------ #
    #  Forward (Level 1: associational prediction)
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]):
        h, mu, logvar, z = self.encode(batch)
        intensity = self.predict_intensity(z, batch)
        return {
            "intensity": intensity,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "h": h,
        }

    # ------------------------------------------------------------------ #
    #  Counterfactual (Level 3)
    # ------------------------------------------------------------------ #
    def counterfactual(
        self,
        batch: Dict[str, torch.Tensor],
        intervention_indices: torch.Tensor,
        intervention_values: torch.Tensor,
    ):
        """Pearl's three-step counterfactual inference.

        Note: intervention_indices index INTO THE LATENT DIMENSION, not the
        node space. For user-level interventions, the demo layer maps a
        user_id to a corresponding latent component (e.g. by hashing) or
        replaces the latent z directly with a zero vector.

        Returns:
            cf_intensity: (B,)
            z_cf:         (B, latent_dim)
        """
        with torch.no_grad():
            _h, _mu, _logvar, z_observed = self.encode(batch)

            # If interventions are over the latent dim, use the SCM directly
            if intervention_indices.numel() > 0:
                z_cf = self.causal_layer.intervene(
                    z_observed, intervention_indices, intervention_values
                )
            else:
                # No interventions: re-run SCM as identity
                z_cf = self.causal_layer.forward(
                    self.causal_layer.abduce(z_observed)
                )

            cf_intensity = self.predict_intensity(z_cf, batch)

        return cf_intensity, z_cf
