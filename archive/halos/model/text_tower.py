"""Phase-4+ extension — a depth-matched text encoder tower + doc cross-attention.

The graph transformer attends to documentation in two optional, config-flagged ways (both default OFF;
the FiLM/pooled-d_e path is unchanged):

* **TextTower** — ``L`` self-attention blocks (L = graph ``n_layers``) over the projected, FROZEN-and-
  CACHED Qwen span memory ``[M, d_text]`` → per-layer doc states ``T^1..T^L`` of ``[M, d_model]``. The
  heavy Qwen pass stays cached; this tower (and the cross-attn) are the trainable "other encoder".
* **DocCrossAttention** — multi-head cross-attention with graph node states as queries and a doc memory
  as keys/values. Used **feature-side** (graph layer ℓ ← ``T^ℓ``) and **geometry-side** (the metapath
  query ← ``T^L`` to build ``ctx(p)`` for g_θ).

Docs are time-independent and blind-authored, so none of this touches scale-equivariance or leakage.
The span memory is shared across the batch (one DB per batch) and ``[0, d_text]`` under the ``null``
regime, in which case every doc-attention contributes exactly zero.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class _EncoderBlock(nn.Module):
    """Pre-norm self-attention + FFN over a set of span tokens."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Linear(ff_mult * d_model, d_model))

    def forward(self, x: Tensor) -> Tensor:                  # x: [1, M, d]
        z = self.norm1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TextTower(nn.Module):
    """Project the cached Qwen span memory and run L self-attention blocks → per-layer doc states."""

    def __init__(self, d_text: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.in_proj = nn.Linear(d_text, d_model)
        self.blocks = nn.ModuleList([_EncoderBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, span_emb: Tensor) -> list[Tensor]:
        """``span_emb`` ``[M, d_text]`` -> list of ``L`` doc states, each ``[M, d_model]``.
        Empty memory (M=0, the null regime) -> L empty states so every cross-attn yields zeros."""
        M = span_emb.shape[0]
        if M == 0:
            empty = span_emb.new_zeros(0, self.d_model)
            return [empty for _ in range(self.n_layers)]
        x = self.in_proj(span_emb.to(self.in_proj.weight.dtype)).unsqueeze(0)   # [1, M, d]
        states = []
        for blk in self.blocks:
            x = blk(x)
            states.append(x.squeeze(0))                                          # [M, d]
        return states


class DocCrossAttention(nn.Module):
    """Cross-attention: queries (graph nodes or a metapath query) attend over a doc memory ``[M, d]``."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, q: Tensor, mem: Tensor) -> Tensor:
        """``q`` ``[B, T, d]``, ``mem`` ``[M, d]`` (shared across batch) -> ``[B, T, d]`` (0 if M=0)."""
        if mem.shape[0] == 0:
            return torch.zeros_like(q)
        B = q.shape[0]
        kv = mem.unsqueeze(0).expand(B, -1, -1)                                  # [B, M, d]
        return self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
