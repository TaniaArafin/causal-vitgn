"""Temporal Graph Encoder using TGN-style memory and attention.


Combines:
 - Per-node memory module (GRU-updated)
 - Functional time encoding (TGAT, Bochner's theorem)
 - Multi-head temporal graph attention


References
----------
Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
  (ICML Workshop on Graph Representation Learning, 2020).
Xu et al., "Inductive Representation Learning on Temporal Graphs"
  (ICLR, 2020).
"""


import math
import torch
import torch.nn as nn




class TimeEncoder(nn.Module):
   """Functional time encoding using learnable Fourier features (TGAT)."""


   def __init__(self, dim: int):
       super().__init__()
       self.dim = dim
       self.linear = nn.Linear(1, dim)
       nn.init.xavier_uniform_(self.linear.weight)


   def forward(self, t: torch.Tensor) -> torch.Tensor:
       # t: (..., 1) — already shaped for Linear projection
       if t.dim() == 0 or t.shape[-1] != 1:
           t = t.unsqueeze(-1)
       return torch.cos(self.linear(t.float()))




class MemoryModule(nn.Module):
   """Per-node memory state s_i(t) updated via GRU on each event."""


   def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
       super().__init__()
       self.num_nodes = num_nodes
       self.memory_dim = memory_dim
       self.gru = nn.GRUCell(message_dim, memory_dim)
       self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
       self.register_buffer("last_update", torch.zeros(num_nodes))


   def reset(self):
       self.memory.zero_()
       self.last_update.zero_()


   def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
       return self.memory[node_ids]


   def update(
       self,
       node_ids: torch.Tensor,
       messages: torch.Tensor,
       t: torch.Tensor,
   ):
       old = self.memory[node_ids]
       new = self.gru(messages, old)
       # Detach memory updates so they don't backprop through previous events
       self.memory[node_ids] = new.detach()
       self.last_update[node_ids] = t.detach()




class TemporalGraphAttention(nn.Module):
   """Multi-head attention pooling over temporal neighbors."""


   def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
       super().__init__()
       self.attn = nn.MultiheadAttention(
           embed_dim, num_heads, dropout=dropout, batch_first=True
       )
       self.norm = nn.LayerNorm(embed_dim)


   def forward(
       self,
       query: torch.Tensor,
       key: torch.Tensor,
       value: torch.Tensor,
       key_padding_mask: torch.Tensor = None,
   ) -> torch.Tensor:
       out, _ = self.attn(
           query, key, value, key_padding_mask=key_padding_mask
       )
       return self.norm(query + out)




class TemporalEncoder(nn.Module):
   """Full temporal graph encoder.


   Pipeline per node:
     memory s_i(t)  -->  temporal attention over recent neighbors  -->  h_i(t)
   """


   def __init__(
       self,
       num_nodes: int,
       memory_dim: int = 128,
       embed_dim: int = 128,
       time_dim: int = 32,
       num_heads: int = 8,
       dropout: float = 0.1,
   ):
       super().__init__()
       self.num_nodes = num_nodes
       self.memory_dim = memory_dim
       self.embed_dim = embed_dim
       self.time_dim = time_dim


       self.time_enc = TimeEncoder(time_dim)


       # Message function: combine src memory + dst memory + time delta
       msg_in_dim = 2 * memory_dim + time_dim
       self.message_net = nn.Sequential(
           nn.Linear(msg_in_dim, msg_in_dim),
           nn.ReLU(),
           nn.Linear(msg_in_dim, memory_dim),
       )


       self.memory = MemoryModule(num_nodes, memory_dim, memory_dim)


       # Project (memory ‖ time_feat) into attention space
       feat_in = memory_dim + time_dim
       self.proj = nn.Linear(feat_in, embed_dim)
       self.attention = TemporalGraphAttention(embed_dim, num_heads, dropout)


   def reset_memory(self):
       self.memory.reset()


   def update_memory(
       self,
       src: torch.Tensor,
       dst: torch.Tensor,
       t: torch.Tensor,
   ):
       """Process a batch of interaction events to update memory."""
       s_src = self.memory.get_memory(src)
       s_dst = self.memory.get_memory(dst)
       dt = (t - self.memory.last_update[src]).clamp(min=0).unsqueeze(-1)
       time_feat = self.time_enc(dt)
       msg_input = torch.cat([s_src, s_dst, time_feat], dim=-1)
       msgs = self.message_net(msg_input)
       self.memory.update(src, msgs, t)


   def forward(
       self,
       target_nodes: torch.Tensor,
       neighbor_nodes: torch.Tensor,
       neighbor_times: torch.Tensor,
       current_time: torch.Tensor,
       neighbor_mask: torch.Tensor = None,
   ) -> torch.Tensor:
       """
       Args:
           target_nodes:   (B,)
           neighbor_nodes: (B, K)
           neighbor_times: (B, K)
           current_time:   (B,)
           neighbor_mask:  (B, K) — True where padded
       Returns:
           h: (B, embed_dim) node embeddings at current_time
       """
       B, K = neighbor_nodes.shape
       device = neighbor_nodes.device


       # Defensive: clamp any stray ids into the valid embedding range.
       # If preprocessing produces correct ids this is a no-op; this guard
       # only fires for unexpected out-of-range ids.
       target_nodes = target_nodes.clamp(0, self.num_nodes - 1)
       neighbor_nodes = neighbor_nodes.clamp(0, self.num_nodes - 1)


       # Target features
       target_mem = self.memory.get_memory(target_nodes)
       target_time_feat = self.time_enc(torch.zeros(B, 1, device=device))
       target_feat = torch.cat([target_mem, target_time_feat], dim=-1)
       target_proj = self.proj(target_feat).unsqueeze(1)  # (B, 1, D)


       # Neighbor features
       nb_mem = self.memory.get_memory(neighbor_nodes.reshape(-1)).view(B, K, -1)
       delta_t = (current_time.unsqueeze(1) - neighbor_times).clamp(min=0).unsqueeze(-1)
       nb_time_feat = self.time_enc(delta_t)
       nb_feat = torch.cat([nb_mem, nb_time_feat], dim=-1)
       nb_proj = self.proj(nb_feat)  # (B, K, D)


       out = self.attention(
           target_proj, nb_proj, nb_proj, key_padding_mask=neighbor_mask
       )
       return out.squeeze(1)

