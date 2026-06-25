"""The SwiGLU FFN and the Mixture-of-Experts FFN (MoRE's only new mechanism).

``MoEFFN`` is a drop-in for the block's ``SwiGLU``: a pool of ``M`` identical SwiGLU experts plus a
sparse top-``k`` router. The crucial asymmetry is enforced by the caller (``rt_substrate``): the router
reads a value-free ``route_feat`` (the relational signature by default) while the experts transform the
evolving hidden state ``x``. Dense expert combine — every expert runs on every token — is the simple,
correct MVP (true sparse dispatch is deferred). Balance is by a router-weight **orthogonality** loss,
not a uniform load-balancing aux loss, so usage may follow the long tail of relation frequencies.
"""
from __future__ import annotations

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
    signature width for ``signature``/``identity``; ``d_model`` for ``hidden``/``value``) — so the
    router input dimension never silently changes across ablation arms.
    """

    def __init__(self, d_model: int, d_ff: int, d_route: int, *, num_experts: int = 4, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        self.router = nn.Linear(d_route, num_experts, bias=False)

    def gates(self, route_feat: Tensor) -> Tensor:
        """-> ``[..., num_experts]`` gate weights; zero off the top-k support, rows sum to 1."""
        logits = self.router(route_feat)                          # [..., E]
        topv, topi = logits.topk(self.k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, topi, topv)
        return F.softmax(masked, dim=-1)

    def forward(self, x: Tensor, route_feat: Tensor) -> tuple[Tensor, Tensor]:
        g = self.gates(route_feat)                                # [..., E]
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                 # dense combine (MVP)
            y = y + g[..., e:e + 1] * expert(x)
        return y, g

    def ortho_loss(self) -> Tensor:
        """``‖ŴŴᵀ − I‖²_F`` over the row-normalized router weights (HOPE-style decorrelation)."""
        W = F.normalize(self.router.weight, dim=-1)               # [E, d_route]
        gram = W @ W.t()                                          # [E, E]
        eye = torch.eye(self.num_experts, device=W.device, dtype=W.dtype)
        return ((gram - eye) ** 2).sum()
