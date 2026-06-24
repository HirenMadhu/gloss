"""Phase 2 — dimensionless time: the scale-equivariance carrier.

``tau`` is always ``log(Δt / T_ctx)`` of a *dimensionless* ratio, so a global clock rescale ``t -> c·t``
(with c>0) cancels (Δt and T_ctx both scale by c) and leaves tau — hence every downstream logit —
invariant. We compute it in float64 and only cast to the model dtype at the boundary.

Two consumers:
* node-time term ``W_t φ(τ_u)`` where ``τ_u = log((seed_time − row_time)/T_ctx)`` (recency of a row
  relative to its seed), here via :func:`node_tau`;
* the pairwise ``τ_uw`` (already in the collate) feeding the bias generator in Phase 3.

``BochnerTime`` are learnable random-Fourier features of tau (Xu et al. 2019), used for the node-time term.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def node_tau(seed_time: Tensor, row_time: Tensor, t_ctx: Tensor, is_timed: Tensor) -> tuple[Tensor, Tensor]:
    """Per-node dimensionless recency ``τ_u = log((seed_time − row_time)/T_ctx)``.

    Returns ``(tau_u [B,N] float64, valid [B,N] bool)``. Valid only for timed nodes with a strictly
    positive gap (seed strictly after the row); timeless rows / zero-gap rows get tau=0, valid=False.
    """
    seed = seed_time.to(torch.float64).view(-1, 1)
    rt = row_time.to(torch.float64)
    tctx = t_ctx.to(torch.float64).view(-1, 1).clamp_min(1e-12)
    delta = seed - rt
    valid = is_timed & (delta > 0)
    tau = torch.zeros_like(rt)
    tau[valid] = torch.log(delta[valid] / tctx.expand_as(rt)[valid])
    return tau, valid


class BochnerTime(nn.Module):
    """Learnable Fourier features of a (dimensionless) time coordinate: φ(τ)_k = √(2/D)·cos(w_k τ + b_k)."""

    def __init__(self, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        # init frequencies on a log-spread so features span several time scales
        self.w = nn.Parameter(torch.logspace(-2, 1, n_freq))
        self.b = nn.Parameter(torch.zeros(n_freq))

    @property
    def out_dim(self) -> int:
        return self.n_freq

    def forward(self, tau: Tensor) -> Tensor:
        """tau ``[...]`` -> features ``[..., n_freq]`` (in the module's dtype)."""
        tau = tau.to(self.w.dtype).unsqueeze(-1)             # [..., 1]
        feats = math.sqrt(2.0 / self.n_freq) * torch.cos(tau * self.w + self.b)
        return feats
