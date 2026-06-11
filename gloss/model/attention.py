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
    def __init__(self, d_model: int, n_heads: int, *, attn_impl: str = "sdpa"):
        super().__init__()
        assert d_model % n_heads == 0
        assert attn_impl in ("sdpa", "flex")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_impl = attn_impl
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
        if self.attn_impl == "flex":
            out = self._flex(q, k, v, gb, geom, log_t_ctx)
        else:
            attn_mask = self._bias(gb, geom, log_t_ctx, h.dtype, h.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # [B, H, N, dh]
        out = out.transpose(1, 2).reshape(B, N, d)
        out = self.o(out)
        return out * gb.pad_mask.to(h.device).unsqueeze(-1)

    def _flex(self, q, k, v, gb: GlossBatch, geom: GeometryTable, log_t_ctx: Tensor | None) -> Tensor:
        """FlexAttention backend: the Gaussian-in-τ bias as a fused ``score_mod`` — no [N,N] bias is
        materialized (the real 'flash for HALOS' at scale). Standard flash-attn can't take this bias.

        NOTE: flex recompiles per distinct N (subgraph size varies), so it's only a win at large N /
        pretraining; SDPA stays the default. Parity with SDPA on real rows is the tested contract.
        """
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        dev = q.device
        a, mu, sigma, b = geom.a.to(dev), geom.mu.to(dev), geom.sigma.to(dev), geom.b.to(dev)
        mp = gb.metapath_id.to(dev)                  # [B, N, N]
        tau = gb.tau.to(dev).to(q.dtype)
        tv = gb.temporal_valid.to(dev)
        attend = gb.attend_mask.to(dev)
        anchor = geom.anchor_w.to(dev) if (geom.anchor_w is not None and log_t_ctx is not None) else None
        ltc = log_t_ctx.to(dev) if log_t_ctx is not None else None

        def score_mod(score, bi, hi, qi, ki):
            p = mp[bi, qi, ki]
            mu_p = mu[p, hi]
            if anchor is not None:
                mu_p = mu_p + anchor[p, hi] * ltc[bi]
            gauss = a[p, hi] * torch.exp(-((tau[bi, qi, ki] - mu_p) ** 2) / (2 * sigma[p, hi] ** 2))
            gauss = torch.where(tv[bi, qi, ki], gauss, torch.zeros_like(gauss))
            return score + b[p, hi] + gauss

        def mask_mod(bi, hi, qi, ki):
            return attend[bi, qi, ki] | (qi == ki)   # keep diagonal so padded rows never fully mask

        Bn, _, N, _ = q.shape
        block_mask = create_block_mask(mask_mod, Bn, None, N, N, device=str(dev))
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
