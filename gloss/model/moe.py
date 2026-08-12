"""The SwiGLU FFN and the Mixture-of-Experts FFN (MoRE's only new mechanism).

``MoEFFN`` is a drop-in for the block's ``SwiGLU``: a pool of ``M`` identical SwiGLU experts plus a
sparse top-``k`` router. The crucial asymmetry is enforced by the caller (``two_level``): the router
reads a value-free ``route_feat`` (the relational signature) while the experts transform the evolving
hidden state ``x``. Balance is by a router-weight **orthogonality** loss, not a uniform load-balancing
aux loss, so usage may follow the long tail of relation frequencies.

**Two combines, identical math.** ``dispatch='dense'`` evaluates every expert on every token and
multiplies the ``M − k`` non-selected ones by a zero gate; ``dispatch='sparse'`` evaluates each expert
only on the tokens routed to it. The *outputs agree* — the gate was always sparse — but the cost does
not: dense is ``M×`` the FFN FLOPs and activations of a plain SwiGLU, sparse is ``k×`` (plus shared).
Dense is the historical default and stays bit-for-bit unchanged, because every measured result in this
repo was produced by it. Sparse is what makes a large expert pool affordable: at ``M=8, k=2`` plus 2
shared it is 4 experts evaluated instead of 10.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: How the top-k mixture is *computed*. Both give the same output; see the module docstring.
DISPATCHES = ("dense", "sparse")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def sparse_combine(
    x: Tensor,
    gates: Tensor,
    topi: Tensor,
    experts: nn.ModuleList,
    *,
    valid: Tensor | None = None,
) -> Tensor:
    """``Σ_{e ∈ top-k} g_e · E_e(x)``, evaluating each expert only on its own tokens.

    ``x`` is ``[..., d]``, ``gates`` ``[..., E]`` (zero off the top-k support), ``topi`` ``[..., k]``
    the selected expert ids. ``valid`` (``[...]`` bool) drops tokens from the computation entirely —
    at the cell level that is padding, and since sampled sequences run 6–20% full at ``seq_len=1024``
    it is a larger saving than the top-k routing itself.

    Equivalent to the dense combine up to summation order when ``valid is None``; with ``valid`` set,
    the excluded positions come back exactly zero instead of carrying an unused mixture.

    One consequence worth stating: an expert that receives **no** tokens contributes nothing to the
    autograd graph, so its parameters get no ``.grad`` on that step. Under DDP that is an unused
    parameter, where the dense combine always produced a (zero-valued) gradient. Callers running DDP
    need ``find_unused_parameters=True`` — which the multi-DB trunk needs anyway, because a batch only
    touches the node types its sample happens to contain.
    """
    d = x.shape[-1]
    n_experts = gates.shape[-1]
    xf = x.reshape(-1, d)
    gf = gates.reshape(-1, n_experts)
    tf = topi.reshape(-1, topi.shape[-1])
    live = valid.reshape(-1) if valid is not None else None

    y = torch.zeros_like(xf)
    for e, expert in enumerate(experts):
        sel = (tf == e).any(-1)
        if live is not None:
            sel = sel & live
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xe = xf.index_select(0, idx)
        ge = gf.index_select(0, idx)[:, e:e + 1]
        # `.to(y.dtype)` is required, not defensive: under autocast the experts return bf16 while
        # `x` (out of an RMSNorm, which autocast does not list) is fp32, and `index_add` demands
        # matching scalar types where the dense path's `+` would have promoted silently. Accumulating
        # the mixture in the wider dtype is also the numerically better choice.
        y = y.index_add(0, idx, (ge * expert(xe)).to(y.dtype))
    return y.view_as(x)


def apply_shared(x: Tensor, shared, *, valid: Tensor | None = None) -> Tensor:
    """Sum of the always-on, ungated shared experts (zero when there are none).

    Shared experts are DeepSeek-MoE style: applied to every token regardless of the gate, so the
    routed experts only have to model what is *specific* to a token's signature rather than each
    re-learning the common transform.
    """
    if not shared:
        return torch.zeros_like(x)
    if valid is None:
        out = shared[0](x)
        for m in shared[1:]:
            out = out + m(x)
        return out
    d = x.shape[-1]
    xf = x.reshape(-1, d)
    idx = valid.reshape(-1).nonzero(as_tuple=True)[0]
    y = torch.zeros_like(xf)
    if idx.numel():
        xe = xf.index_select(0, idx)
        acc = shared[0](xe)
        for m in shared[1:]:
            acc = acc + m(xe)
        y = y.index_add(0, idx, acc.to(y.dtype))     # see sparse_combine: autocast dtype mismatch
    return y.view_as(x)


class MoEFFN(nn.Module):
    """Pool of ``num_experts`` SwiGLU experts; sparse top-``k`` gate on ``route_feat``.

    ``d_route`` is the width of the routing feature, fixed by the caller — the signature width for
    the ``signature`` arm. Stating it explicitly is what keeps the router input dimension from
    silently changing when the routing feature changes.

    ``num_shared`` adds always-on ungated experts alongside the routed pool. It defaults to 0, which
    is the measured cell-level configuration: the ``+S`` arm was only mildly positive, on regression
    alone, so the cell level is routed-only while ``RowMoE`` is shared+routed. Pretraining-scale
    configurations override it, and that is a deliberate deviation rather than drift.

    The S/C/P/H ablation additions (cosine router, top-p support, hierarchical gate) are retired to
    ``archive/multi-level/model/moe_ablations.py``.
    """

    def __init__(self, d_model: int, d_ff: int, d_route: int, *, num_experts: int = 4, k: int = 2,
                 num_shared: int = 0, dispatch: str = "dense"):
        super().__init__()
        if dispatch not in DISPATCHES:
            raise ValueError(f"unknown dispatch {dispatch!r}; expected one of {DISPATCHES}")
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.num_shared = num_shared
        self.dispatch = dispatch
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        self.shared = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_shared)) or None
        self.router = nn.Linear(d_route, num_experts, bias=False)

    def _gates_topi(self, route_feat: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(route_feat)
        topv, topi = logits.topk(self.k, dim=-1)
        # `masked` is a fresh constant tensor, so the -inf entries carry no gradient: the router only
        # ever learns through the k scattered logits. That is why the sparse combine below reproduces
        # the dense one's gradients exactly and not merely its forward.
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, topi, topv)
        return F.softmax(masked, dim=-1), topi

    def gates(self, route_feat: Tensor) -> Tensor:
        """-> ``[..., num_experts]`` gate weights; zero off the top-k support, rows sum to 1."""
        return self._gates_topi(route_feat)[0]

    def forward(self, x: Tensor, route_feat: Tensor, *,
                valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """-> ``(y, gates)``. ``valid`` is honoured only by ``dispatch='sparse'``.

        Dense deliberately ignores ``valid``: it is the path every reported number was measured on,
        and masking it would change those numbers at padding positions. Nothing downstream reads a
        padding cell's state (cell attention masks it, ``RowPool`` groups exclude it, the head reads
        row tokens), so the two paths differ only where the result is dead.
        """
        g, topi = self._gates_topi(route_feat)
        if self.dispatch == "sparse":
            y = sparse_combine(x, g, topi, self.experts, valid=valid)
            if self.shared:
                y = y + apply_shared(x, self.shared, valid=valid)
            return y, g
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                         # dense combine
            y = y + g[..., e:e + 1] * expert(x)
        if self.shared:
            y = y + apply_shared(x, self.shared)
        return y, g

    def ortho_loss(self) -> Tensor:
        """``‖ŴŴᵀ − I‖²_F`` over the row-normalized router directions."""
        W = F.normalize(self.router.weight, dim=-1)                       # [E, d_route]
        gram = W @ W.t()                                                  # [E, E]
        eye = torch.eye(self.num_experts, device=W.device, dtype=W.dtype)
        return ((gram - eye) ** 2).sum()
