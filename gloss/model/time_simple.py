"""Simple temporal bias for Paper 1 (implementation.md §5). Phase-3 STUB.

    B_time(i,j) = -alpha_head * log(1 + dt_ij)     # alpha_head >= 0 (softplus), per head
    # or RT's normalized datetime scalar; temporal.mode in {log_decay, rt_scalar}.

This is NOT the deferred content-addressed Hawkes kernel (that is gloss/ext, Paper #2). Do not build
before GATE 1.
"""
from __future__ import annotations


class SimpleTemporalBias:
    """Per-head monotone log-dt decay. forward(dt ``[..., n, n]`` seconds) -> bias ``[H, n, n]``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 — build after GATE 1.")
