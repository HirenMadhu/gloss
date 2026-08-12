"""Retired MoE ablation arms (the v2 S/C/P/H era, superseded).

S = shared always-on expert, C = cosine router over learnable keys, P = top-p adaptive
support, H = hierarchical two-level gate. Reachable only through `scripts/run_ablation.py`,
never through `run_gridsearch.py`, and no multi-level result used any of them.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gloss.model.moe import SwiGLU  # noqa: F401


class HMoEFFN(nn.Module):
    """Hierarchical two-level MoE FFN (**H**): a learned soft gate over ``n_groups`` groups, then a
    per-group top-``k2`` gate over that group's experts, dense-combined:

        y = Σ_g G¹_g · Σ_{j∈group_g} G²_{g,j} · E_{g,j}(x)

    Reads the same value-free ``route_feat`` as :class:`MoEFFN`. Balance = level-1 gate-row orthogonality
    (static, param-based) **plus** a level-1 occupancy term ``log(Γ) − H(mean G¹)`` (≥ 0, zero when the
    groups are used uniformly) that resists group collapse. Dense combine only (top-1-group sparse dispatch
    is deferred). Returns ``(y, G¹)``; the block discards the gates and reads only ``ortho_loss()``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_route: int,
        *,
        n_groups: int = 4,
        experts_per_group: int = 2,
        k2: int = 1,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.experts_per_group = experts_per_group
        self.k2 = min(k2, experts_per_group)
        self.g1 = nn.Linear(d_route, n_groups, bias=False)                # level-1: over groups
        self.g2 = nn.ModuleList(nn.Linear(d_route, experts_per_group, bias=False)
                                for _ in range(n_groups))                 # level-2: within each group
        self.experts = nn.ModuleList(
            nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(experts_per_group))
            for _ in range(n_groups))
        self._balance: Tensor | float = 0.0                               # set each forward (level-1 occupancy)

    def forward(self, x: Tensor, route_feat: Tensor) -> tuple[Tensor, Tensor]:
        p1 = F.softmax(self.g1(route_feat), dim=-1)                       # [..., G] soft over groups
        y = torch.zeros_like(x)
        for gi in range(self.n_groups):
            l2 = self.g2[gi](route_feat)
            tv, ti = l2.topk(self.k2, dim=-1)                             # within-group top-k2
            m = torch.full_like(l2, float("-inf"))
            m.scatter_(-1, ti, tv)
            p2 = F.softmax(m, dim=-1)
            grp = torch.zeros_like(x)
            for e, expert in enumerate(self.experts[gi]):
                grp = grp + p2[..., e:e + 1] * expert(x)
            y = y + p1[..., gi:gi + 1] * grp
        occ = p1.reshape(-1, self.n_groups).mean(0)                       # [G] mean group occupancy
        entropy = -(occ.clamp_min(1e-9) * occ.clamp_min(1e-9).log()).sum()
        self._balance = math.log(self.n_groups) - entropy                # >= 0, 0 when uniform
        return y, p1

    def ortho_loss(self) -> Tensor:
        """Level-1 gate-row decorrelation + the last forward's level-1 occupancy penalty."""
        W = F.normalize(self.g1.weight, dim=-1)                           # [G, d_route]
        gram = W @ W.t()
        eye = torch.eye(self.n_groups, device=W.device, dtype=W.dtype)
        return ((gram - eye) ** 2).sum() + self._balance
