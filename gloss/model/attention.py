"""RelationalAttention (implementation.md §4.3, §5). Phase-3 STUB.

    logits(i,j) = (Q_i . K_j)/sqrt(d) + B_time(i,j) + B_hop(i,j)   # masked by attn_mask_kind

RT masks {column, feature, neighbor, full}; full OFF by default. **Memory (plan stress-test #5):** do NOT
materialize a dense [B,H,T,T] bias at max_cells=4096 — use blocked/structured attention per mask-kind over
neighbor sets, with additive B_time+B_hop within blocks. Backend = PyTorch SDPA (mem-efficient); flash-attn
only for unbiased paths.
"""
from __future__ import annotations


class RelationalAttention:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — build after GATE 1; honor the blocked-attention constraint.")
