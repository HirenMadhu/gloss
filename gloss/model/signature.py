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


class RelationalSignature(nn.Module):
    """The value-free cell routing signature: name + modality + recency.

    Recency is the continuous :class:`~gloss.model.time_encoding.TimeLadder` read out as
    ``[sin θ ; cos θ]`` behind a learned ``W_τ``. The 20 fixed log-decade buckets it replaced are
    retired with the phase-0 ladder — 14 of the 20 were empty on rel-f1.

    The signature stays **value-free** and invariant to which neighbours were sampled: τ depends only
    on the cell's own timestamp and the seed time, and ω is a fixed non-learned constant. That is
    what makes routing leak-free by construction (`test_routing_invariance.py`).
    """

    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, *, d_sig: int = 128,
                 ladder=None):
        super().__init__()
        from .time_encoding import TimeLadder

        d_text = int(name_emb.shape[1])
        self.d_sig = d_sig
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.schema_proj = nn.Linear(d_text, d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        self.ladder = ladder or TimeLadder()
        self.w_tau = nn.Linear(self.ladder.feat_dim, d_sig, bias=False)
        self.norm = nn.RMSNorm(d_sig)

    @torch.no_grad()
    def column_signature(self) -> Tensor:
        """-> ``[C, d_sig]``: every column's signature at the **untimed** recency state.

        The per-column half of ``forward`` with the time term held at its context-independent
        "unknown Δ" value — bin 0 under ``buckets``, the all-zero sinusoid pair under ``rope``.
        Diagnostics use it to ask which expert a *column* routes to without a batch, and it must be
        derived here rather than reconstructed by a caller.
        """
        name = self.name_emb                                                 # [C, d_text]
        z = self.schema_proj(name) + self.stype_emb(self.modality_id)        # [C, d_sig]
        untimed = torch.zeros(name.shape[0], dtype=torch.bool, device=name.device)
        tau = torch.zeros(name.shape[0], device=name.device)
        z = z + self.w_tau(self.ladder.feats(tau, untimed, untimed).to(z.dtype))
        return self.norm(z)

    def forward(self, cb: CellBatch) -> Tensor:
        col = cb.col_idxs.clamp(min=0)                                       # [B, S] (pad -1 -> 0)
        B, S = col.shape
        flat = col.reshape(-1)
        name = self.name_emb.index_select(0, flat).view(B, S, -1)           # [B, S, d_text]
        mod = self.modality_id.index_select(0, flat).view(B, S)             # [B, S]
        z = self.schema_proj(name) + self.stype_emb(mod)
        # Untimed cells get an all-zero feature pair (not [0;1]), so "unknown" stays linearly
        # separable from "Delta = 0" behind W_tau.
        seed_t = cb.seed_time.unsqueeze(1)
        tau = self.ladder.tau_from_times(seed_t, cb.row_time)
        clamped = self.ladder.was_clamped(seed_t, cb.row_time) & cb.is_timed
        z = z + self.w_tau(self.ladder.feats(tau, cb.is_timed, clamped).to(z.dtype))
        return self.norm(z)                                                 # [B, S, d_sig]
