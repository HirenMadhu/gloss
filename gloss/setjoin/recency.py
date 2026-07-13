"""Fixed log-spaced Δt recency buckets (standalone transplant of the scheme in model/signature.py).

Bucket boundaries are **fixed and context-independent** — one bucket per order of magnitude of
Δ = seed_time − row_time in the native time unit — so a row's bin depends only on its own timestamp and
the seed time, never on the batch or the sampled neighborhood. Bin 0 is reserved for untimed/pad.
"""
from __future__ import annotations

import torch
from torch import Tensor

RECENCY_EDGES = [10.0 ** i for i in range(0, 19)]     # 1, 10, ..., 1e18
N_RECENCY_BINS = len(RECENCY_EDGES) + 2               # 0 = untimed/pad, 1..len(edges)+1 = timed buckets


def recency_bins(seed_time: Tensor, row_time: Tensor, is_timed: Tensor) -> Tensor:
    """-> long bins shaped like ``row_time``; 0 for untimed, else 1 + the order-of-magnitude bucket.

    ``seed_time`` is ``[B]`` and broadcasts against ``row_time``/``is_timed`` of shape ``[B, ...]``.
    """
    edges = torch.tensor(RECENCY_EDGES, dtype=torch.float32, device=row_time.device)
    st = seed_time
    while st.dim() < row_time.dim():
        st = st.unsqueeze(-1)
    dt = (st - row_time).clamp(min=0.0).to(torch.float32)
    bins = torch.bucketize(dt, edges) + 1
    return torch.where(is_timed, bins, torch.zeros_like(bins)).long()
