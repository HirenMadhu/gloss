"""The SwiGLU FFN and the Mixture-of-Experts FFN (MoRE's only new mechanism).

``MoEFFN`` is a drop-in for the block's ``SwiGLU``: a pool of ``M`` identical SwiGLU experts plus a
sparse top-``k`` router. The crucial asymmetry is enforced by the caller (``rt_substrate``): the router
reads a value-free ``route_feat`` (the relational signature by default) while the experts transform the
evolving hidden state ``x``. Dense expert combine — every expert runs on every token — is the simple,
correct MVP (true sparse dispatch is deferred). Balance is by a router-weight **orthogonality** loss,
not a uniform load-balancing aux loss, so usage may follow the long tail of relation frequencies.

Beyond the base top-k router, ``MoEFFN`` carries three optional ablation additions (all off by default,
so the base arm is unchanged): a **shared** always-on expert (S), a **cosine**/normalized router (C), and
**Top-P** adaptive expert count (P). ``HMoEFFN`` is the fourth, hierarchical addition (H) — a two-level
gate — kept as a separate class so the flat ``MoEFFN`` stays simple.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Pool of ``num_experts`` SwiGLU experts; sparse top-``k`` gate on ``route_feat``.

    ``d_route`` is the width of the routing feature and is fixed by the caller's routing mode (the
    signature width for ``signature``/``identity``; ``d_model`` for ``hidden``/``value``; ``d_sig+d_model``
    for ``hybrid``) — so the router input dimension never silently changes across ablation arms.

    Ablation additions (default off = base top-k router):
      * ``use_shared`` (**S**) — an always-on shared expert added to the routed sum.
      * ``cosine`` (**C**) — cosine-normalized logits over learnable ``keys`` at temperature ``tau``
        (``ortho_loss`` then decorrelates the keys instead of the linear router weights).
      * ``top_p`` (**P**) — if set, replace the fixed top-k support with the smallest set of experts whose
        softmax mass reaches ``top_p`` (adaptive per-token expert count).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_route: int,
        *,
        num_experts: int = 4,
        k: int = 2,
        use_shared: bool = False,
        cosine: bool = False,
        tau: float = 0.3,
        top_p: float | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.cosine = cosine
        self.tau = tau
        self.top_p = top_p
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        if cosine:
            self.keys = nn.Parameter(torch.randn(num_experts, d_route))   # C: learnable expert keys
        else:
            self.router = nn.Linear(d_route, num_experts, bias=False)
        self.shared = SwiGLU(d_model, d_ff) if use_shared else None       # S: always-on expert

    def _logits(self, route_feat: Tensor) -> Tensor:
        if self.cosine:
            rc = F.normalize(route_feat, dim=-1)
            kc = F.normalize(self.keys, dim=-1)
            return (rc @ kc.t()) / self.tau                               # [..., E]
        return self.router(route_feat)

    def gates(self, route_feat: Tensor) -> Tensor:
        """-> ``[..., num_experts]`` gate weights; zero off the (top-k or top-p) support, rows sum to 1."""
        logits = self._logits(route_feat)
        if self.top_p is None:                                            # top-k (base behavior)
            topv, topi = logits.topk(self.k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topi, topv)
            return F.softmax(masked, dim=-1)
        probs = F.softmax(logits, dim=-1)                                 # P: adaptive top-p support
        sp, idx = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < self.top_p                          # smallest prefix reaching P (>=1)
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, idx, keep)
        g = probs.masked_fill(~mask, 0.0)
        return g / g.sum(-1, keepdim=True).clamp_min(1e-9)

    def forward(self, x: Tensor, route_feat: Tensor) -> tuple[Tensor, Tensor]:
        g = self.gates(route_feat)                                        # [..., E]
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                         # dense combine (MVP)
            y = y + g[..., e:e + 1] * expert(x)
        if self.shared is not None:
            y = y + self.shared(x)                                        # S: residual over the shared expert
        return y, g

    def ortho_loss(self) -> Tensor:
        """``‖ŴŴᵀ − I‖²_F`` over the row-normalized router directions (keys if cosine, else router weight)."""
        W = F.normalize(self.keys if self.cosine else self.router.weight, dim=-1)   # [E, d_route]
        gram = W @ W.t()                                                  # [E, E]
        eye = torch.eye(self.num_experts, device=W.device, dtype=W.dtype)
        return ((gram - eye) ** 2).sum()


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
