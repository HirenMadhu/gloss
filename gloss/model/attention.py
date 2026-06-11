"""Phase 3 — relational attention with the doc-generated Gaussian-in-τ bias.

For a pair (i attends j) linked by metapath ``p``, with dimensionless lag ``τ_ij``:

    B_h(i,j) = b_h(p)  +  temporal_valid · a_h(p) · exp( −(τ_ij − μ_h(p))² / (2 σ_h(p)²) )

``b_h`` is the structural bucket (applies to every attendable pair, including timeless / zero-gap /
>1-hop pairs); the Gaussian term is added only where both endpoints are timestamped. The bias is fed to
``F.scaled_dot_product_attention`` as an additive float mask (its memory-efficient kernel supports an
arbitrary additive bias, which flash-attn's API does not — hence SDPA here). Non-attendable pairs get
``-inf``; the diagonal is always kept finite so padded rows never produce a NaN softmax.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.collate import GlossBatch
from .bias_generator import GeometryTable

NEG_INF = float("-inf")


class RelationalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def _bias(self, gb: GlossBatch, geom: GeometryTable, log_t_ctx: Tensor | None, dtype, device) -> Tensor:
        """Additive attention mask ``[B, H, N, N]`` (bias where attendable, −inf elsewhere)."""
        mp = gb.metapath_id.to(device)                 # [B, N, N] in [0, P)
        a = geom.a[mp]                                  # [B, N, N, H]
        mu = geom.mu[mp]
        sigma = geom.sigma[mp]
        b = geom.b[mp]
        tau = gb.tau.to(dtype).to(device).unsqueeze(-1)        # [B, N, N, 1]
        tv = gb.temporal_valid.to(device).unsqueeze(-1)        # [B, N, N, 1]

        if geom.anchor_w is not None and log_t_ctx is not None:
            aw = geom.anchor_w[mp]                              # [B, N, N, H]
            mu = mu + aw * log_t_ctx.view(-1, 1, 1, 1)         # absolute-scale shift (breaks invariance)

        gauss = a * torch.exp(-((tau - mu) ** 2) / (2 * sigma ** 2))
        bias = b + torch.where(tv, gauss, torch.zeros_like(gauss))   # [B, N, N, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()                # [B, H, N, N]

        attend = gb.attend_mask.to(device).unsqueeze(1)             # [B, 1, N, N]
        neg = torch.full_like(bias, NEG_INF)
        mask = torch.where(attend, bias, neg)
        # keep the diagonal finite so fully-padded rows don't NaN the softmax
        N = mask.shape[-1]
        eye = torch.eye(N, dtype=torch.bool, device=device).view(1, 1, N, N)
        mask = torch.where(eye, bias, mask)
        return mask

    def forward(self, h: Tensor, gb: GlossBatch, geom: GeometryTable, *, log_t_ctx: Tensor | None = None) -> Tensor:
        B, N, d = h.shape
        H, dh = self.n_heads, self.d_head

        def split(x):
            return x.view(B, N, H, dh).transpose(1, 2)            # [B, H, N, dh]

        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        attn_mask = self._bias(gb, geom, log_t_ctx, h.dtype, h.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # [B, H, N, dh]
        out = out.transpose(1, 2).reshape(B, N, d)
        out = self.o(out)
        return out * gb.pad_mask.to(h.device).unsqueeze(-1)
