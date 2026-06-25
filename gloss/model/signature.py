"""The value-free per-cell **relational signature** the MoE router routes on.

For each cell the signature is

    z_{u,c} = RMSNorm( W_s · name_emb[c]  +  ψ(modality[c])  +  φ(recency_bin(u)) )

— the frozen-LM column-name embedding (schema/semantic axis), a learned modality embedding (the
pytorch-frame stype: numeric / categorical / text / timestamp), and a learned embedding of the cell's
**causal recency** Δ = seed_time − row_time. Every term is measurable from the cell's own ``(column,
modality, timestamp)`` and the seed time; **nothing about the sampled neighborhood enters z**, which is
exactly what makes signature routing temporally leak-free and invariant to which neighbors were drawn
(see ``tests/test_routing_invariance.py``). z is computed once in the trunk and reused at every block.

Recency uses **fixed, context-independent** order-of-magnitude buckets (so the bins do not depend on the
batch — a batch-relative quantile would reintroduce a dependence on the neighbors). Bin 0 is reserved
for untimed / pad cells.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import CellBatch

# Fixed recency boundaries: one bucket per order of magnitude in the native time unit. Robust to the
# unit (seconds / days / nanoseconds) up to a constant shift in which buckets get used.
_RECENCY_EDGES = [10.0 ** i for i in range(0, 19)]   # 1, 10, ..., 1e18


class RelationalSignature(nn.Module):
    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, *, d_sig: int = 128):
        super().__init__()
        d_text = int(name_emb.shape[1])
        self.d_sig = d_sig
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.register_buffer("recency_edges", torch.tensor(_RECENCY_EDGES, dtype=torch.float32),
                             persistent=False)
        self.n_recency = len(_RECENCY_EDGES) + 1                 # timed buckets 0..len(edges)
        self.schema_proj = nn.Linear(d_text, d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        self.recency_emb = nn.Embedding(self.n_recency + 1, d_sig)   # +1: index 0 == untimed/pad
        self.norm = nn.RMSNorm(d_sig)

    def recency_bins(self, cb: CellBatch) -> Tensor:
        """-> ``[B, S]`` long; bin 0 for untimed/pad, else 1 + the order-of-magnitude bucket of Δ."""
        dt = (cb.seed_time.unsqueeze(1) - cb.row_time).clamp(min=0.0).to(torch.float32)   # [B, S]
        bucket = torch.bucketize(dt, self.recency_edges.to(dt.device))      # 0..len(edges)
        bins = bucket + 1                                                   # 1..n_recency
        return torch.where(cb.is_timed, bins, torch.zeros_like(bins)).long()

    def forward(self, cb: CellBatch) -> Tensor:
        col = cb.col_idxs.clamp(min=0)                                       # [B, S] (pad -1 -> 0)
        B, S = col.shape
        flat = col.reshape(-1)
        name = self.name_emb.index_select(0, flat).view(B, S, -1)           # [B, S, d_text]
        mod = self.modality_id.index_select(0, flat).view(B, S)             # [B, S]
        rec = self.recency_bins(cb)                                         # [B, S]
        z = self.schema_proj(name) + self.stype_emb(mod) + self.recency_emb(rec)
        return self.norm(z)                                                 # [B, S, d_sig]
