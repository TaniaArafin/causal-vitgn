"""Smoke tests for model components.


Run:
   PYTHONPATH=. python tests/test_models.py
"""


import torch


from src.models.temporal_encoder import TemporalEncoder, TimeEncoder
from src.models.variational_encoder import (
   CascadeContextEncoder,
   VariationalEncoder,
)
from src.models.hawkes_decoder import NeuralHawkesDecoder
from src.models.causal_layer import StructuralCausalLayer
from src.models.causal_vitgn import CausalVITGN




# ----------------------------------------------------------------------- #
#  Day 2 tests
# ----------------------------------------------------------------------- #
def test_time_encoder():
   enc = TimeEncoder(dim=16)
   t = torch.rand(8, 1)
   out = enc(t)
   assert out.shape == (8, 16), f"expected (8, 16), got {out.shape}"
   print("OK  TimeEncoder")




def test_temporal_encoder():
   num_nodes = 100
   encoder = TemporalEncoder(
       num_nodes=num_nodes,
       memory_dim=32,
       embed_dim=32,
       time_dim=8,
       num_heads=2,
       dropout=0.1,
   )
   encoder.reset_memory()


   B, K = 4, 5
   target = torch.randint(0, num_nodes, (B,))
   nb_nodes = torch.randint(0, num_nodes, (B, K))
   nb_times = torch.rand(B, K) * 100
   cur_t = torch.rand(B) * 100 + 100


   h = encoder(target, nb_nodes, nb_times, cur_t)
   assert h.shape == (B, 32), f"expected (B, 32), got {h.shape}"
   print("OK  TemporalEncoder")




def test_variational_encoder():
   embed_dim, ctx_dim, latent_dim = 32, 32, 16
   ctx_enc = CascadeContextEncoder(embed_dim, ctx_dim)
   var_enc = VariationalEncoder(embed_dim, ctx_dim, latent_dim)


   B, M = 4, 6
   target = torch.randn(B, embed_dim)
   activated = torch.randn(B, M, embed_dim)


   ctx = ctx_enc(target, activated)
   assert ctx.shape == (B, ctx_dim), f"context shape wrong: {ctx.shape}"


   mu, logvar = var_enc(target, ctx)
   assert mu.shape == (B, latent_dim)
   assert logvar.shape == (B, latent_dim)


   z = var_enc.reparameterize(mu, logvar)
   assert z.shape == (B, latent_dim)


   kl = var_enc.kl_divergence(mu, logvar)
   assert kl.shape == (B,)
   print("OK  VariationalEncoder")




def test_hawkes_decoder():
   latent_dim, time_dim = 16, 8
   decoder = NeuralHawkesDecoder(latent_dim, time_dim)


   B, K = 4, 5
   z_user = torch.randn(B, latent_dim)
   dt = torch.rand(B) * 10
   z_neighbors = torch.randn(B, K, latent_dim)
   nb_times = torch.rand(B, K) * 100
   cur_t = torch.rand(B) * 100 + 100


   logits = decoder(z_user, dt, z_neighbors, nb_times, cur_t)
   assert logits.shape == (B,)
   assert torch.isfinite(logits).all(), "Decoder logits must be finite"
   print("OK  NeuralHawkesDecoder")




# ----------------------------------------------------------------------- #
#  Day 3 tests
# ----------------------------------------------------------------------- #
def test_structural_causal_layer():
   latent_dim = 16
   scm = StructuralCausalLayer(latent_dim=latent_dim)


   B = 4
   z = torch.randn(B, latent_dim)


   # Forward: z = (I - A)^{-1} eps
   eps = scm.abduce(z)
   z_round = scm.forward(eps)
   assert z_round.shape == z.shape
   # Round-trip identity (no intervention) should reconstruct z exactly.
   assert torch.allclose(z_round, z, atol=1e-4), \
       f"Round-trip failed: max diff = {(z_round - z).abs().max()}"


   # DAG and sparsity penalties are scalars
   h_A = scm.get_dag_constraint()
   sp = scm.get_sparsity_penalty()
   assert h_A.dim() == 0 and sp.dim() == 0


   print("OK  StructuralCausalLayer (round-trip identity)")




def test_intervention():
   latent_dim = 16
   scm = StructuralCausalLayer(latent_dim=latent_dim)


   B = 4
   z = torch.randn(B, latent_dim)


   # Intervene on latent component 5: set to 0
   idx = torch.tensor([5], dtype=torch.long)
   val = torch.zeros(1, dtype=z.dtype)  # same scalar broadcast for each batch


   # Need the right shape for our `_force_values` (B, k) for 2D z
   val = torch.zeros(B, 1)
   z_cf = scm.intervene(z, idx, val)


   assert z_cf.shape == z.shape
   assert torch.allclose(z_cf[:, 5], torch.zeros(B), atol=1e-6), \
       "Intervened component should equal x'"
   print("OK  StructuralCausalLayer (intervention forces value)")




def test_causal_vitgn_forward():
   config = {
       "model": {
           "num_nodes": 100,
           "memory_dim": 32,
           "embedding_dim": 32,
           "latent_dim": 16,
           "time_encoding_dim": 8,
           "num_heads": 2,
           "dropout": 0.1,
       },
   }
   model = CausalVITGN(config)
   model.temporal_encoder.reset_memory()


   B, K, M = 4, 5, 3
   batch = {
       "target_nodes": torch.randint(0, 100, (B,)),
       "neighbor_nodes": torch.randint(0, 100, (B, K)),
       "neighbor_times": torch.rand(B, K) * 100,
       "current_time": torch.rand(B) * 100 + 100,
       "activated_embs": torch.randn(B, M, 32),
       "z_neighbors": torch.randn(B, K, 16),
       "time_since_last": torch.rand(B) * 10,
   }
   out = model(batch)
   assert out["intensity"].shape == (B,)
   assert out["mu"].shape == (B, 16)
   assert out["logvar"].shape == (B, 16)
   assert out["z"].shape == (B, 16)
   print("OK  CausalVITGN.forward")




def test_causal_vitgn_counterfactual():
   config = {
       "model": {
           "num_nodes": 100,
           "memory_dim": 32,
           "embedding_dim": 32,
           "latent_dim": 16,
           "time_encoding_dim": 8,
           "num_heads": 2,
           "dropout": 0.1,
       },
   }
   model = CausalVITGN(config)
   model.temporal_encoder.reset_memory()


   B, K, M = 4, 5, 3
   batch = {
       "target_nodes": torch.randint(0, 100, (B,)),
       "neighbor_nodes": torch.randint(0, 100, (B, K)),
       "neighbor_times": torch.rand(B, K) * 100,
       "current_time": torch.rand(B) * 100 + 100,
       "activated_embs": torch.randn(B, M, 32),
       "z_neighbors": torch.randn(B, K, 16),
       "time_since_last": torch.rand(B) * 10,
   }


   # Intervene on latent components 0 and 3: clamp them to zero.
   # `z` is (B, latent_dim) so values are 1D of length k.
   idx = torch.tensor([0, 3], dtype=torch.long)
   val = torch.zeros(2)
   cf_int, z_cf = model.counterfactual(batch, idx, val)
   assert cf_int.shape == (B,)
   assert z_cf.shape == (B, 16)
   assert torch.allclose(z_cf[:, 0], torch.zeros(B), atol=1e-6)
   assert torch.allclose(z_cf[:, 3], torch.zeros(B), atol=1e-6)
   print("OK  CausalVITGN.counterfactual")




def main():
   print("Running smoke tests...\n")
   print("--- Day 2 ---")
   test_time_encoder()
   test_temporal_encoder()
   test_variational_encoder()
   test_hawkes_decoder()
   print("\n--- Day 3 ---")
   test_structural_causal_layer()
   test_intervention()
   test_causal_vitgn_forward()
   test_causal_vitgn_counterfactual()
   print("\nAll smoke tests passed.")




if __name__ == "__main__":
   main()





