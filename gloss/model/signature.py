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
    """``time_mode`` selects the recency term (changes.md §3.1 / §3.3):

    * ``buckets`` (default) — the 20 fixed log-decade bins above. **Phase 0a requires this**, so it
      cannot be deleted yet despite §3.1 also saying "delete ``_RECENCY_EDGES``"; the two statements
      contradict each other and this is the resolution.
    * ``rope`` — the continuous :class:`~gloss.model.time_encoding.TimeLadder` read out as
      ``[sin θ ; cos θ]`` behind a learned ``W_τ``. Phase 1 flips to this.

    Both keep the signature **value-free** and invariant to which neighbours were sampled: τ depends
    only on the cell's own timestamp and the seed time, and ω is a fixed constant.
    """

    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, *, d_sig: int = 128,
                 time_mode: str = "buckets", ladder=None):
        super().__init__()
        if time_mode not in ("buckets", "rope"):
            raise ValueError(f"unknown time_mode {time_mode!r}")
        d_text = int(name_emb.shape[1])
        self.d_sig = d_sig
        self.time_mode = time_mode
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.register_buffer("recency_edges", torch.tensor(_RECENCY_EDGES, dtype=torch.float32),
                             persistent=False)
        self.n_recency = len(_RECENCY_EDGES) + 1                 # timed buckets 0..len(edges)
        self.schema_proj = nn.Linear(d_text, d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        if time_mode == "buckets":
            self.recency_emb = nn.Embedding(self.n_recency + 1, d_sig)   # +1: index 0 == untimed/pad
        else:
            from .time_encoding import TimeLadder

            self.ladder = ladder or TimeLadder()
            self.w_tau = nn.Linear(self.ladder.feat_dim, d_sig, bias=False)
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
        z = self.schema_proj(name) + self.stype_emb(mod)
        if self.time_mode == "buckets":
            z = z + self.recency_emb(self.recency_bins(cb))
        else:
            # 14 of the 20 buckets were empty; the ladder spends every channel instead. Untimed cells
            # get an all-zero feature pair (not [0;1]), so "unknown" stays linearly separable from
            # "Delta = 0" behind W_tau.
            seed_t = cb.seed_time.unsqueeze(1)
            tau = self.ladder.tau_from_times(seed_t, cb.row_time)
            clamped = self.ladder.was_clamped(seed_t, cb.row_time) & cb.is_timed
            z = z + self.w_tau(self.ladder.feats(tau, cb.is_timed, clamped).to(z.dtype))
        return self.norm(z)                                                 # [B, S, d_sig]
