"""Diagnostics for the MoE: expert-usage entropy and the cross-table specialization probe.

These are the non-degeneracy / type-overfitting evidence from the method's claims:
- ``expert_usage`` — does the gate spread mass across experts (entropy not collapsed), tracking the
  long tail rather than flattening to uniform?
- ``specialization_probe`` — with ``signature`` routing, do semantically-similar columns *across tables*
  map to the same expert (vs an identity router that just partitions by table)?
"""
from __future__ import annotations

import torch

from ..model.moe import MoEFFN


@torch.no_grad()
def expert_usage(model, cell_batches) -> tuple[torch.Tensor, float]:
    """Mean per-expert gate mass over ``cell_batches`` (an iterable of CellBatch already on the model's
    device) -> ``(usage [E], entropy)``. Hooks every ``MoEFFN``; returns ``(None, nan)`` for dense models."""
    moes = [m for m in model.modules() if isinstance(m, MoEFFN)]
    if not moes:
        return None, float("nan")
    acc = torch.zeros(moes[0].num_experts)

    def hook(_mod, _inp, out):
        _y, g = out
        acc.add_(g.detach().flatten(0, -2).sum(0).cpu())

    handles = [m.register_forward_hook(hook) for m in moes]
    was_training = model.training
    model.eval()
    try:
        for cb in cell_batches:
            model(cb)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    usage = acc / acc.sum().clamp_min(1e-9)
    entropy = float(-(usage.clamp_min(1e-9) * usage.clamp_min(1e-9).log()).sum())
    return usage, entropy


@torch.no_grad()
def specialization_probe(model) -> dict[int, int]:
    """-> ``{column_id: expert}``: the argmax expert each column routes to under the first MoE block's
    router, at recency bin 0 (**signature arm only**; ``{}`` otherwise). Cluster columns by value to see
    cross-table semantic groups. Uses the MoE's own ``_logits`` so cosine and linear routers both work."""
    if getattr(model, "route_on", None) != "signature":      # hybrid/hidden route depend on h -> ill-defined here
        return {}
    sig = getattr(model, "signature", None)
    if sig is None:
        return {}
    moe = next((m for m in model.substrate.modules() if isinstance(m, MoEFFN)), None)
    if moe is None:
        return {}
    # Ask the signature for the column table rather than rebuilding it here: this used to inline
    # `sig.recency_emb(zeros)`, which only exists under time_mode="buckets" and raised AttributeError
    # under "rope" — the mode the two-level runs actually use, so the probe was dead on arrival.
    z = sig.column_signature()                                # [C, d_sig]
    expert = moe._logits(z).argmax(dim=-1)                    # [C]
    return {i: int(expert[i]) for i in range(int(z.shape[0]))}


@torch.no_grad()
def mean_active_experts(model, cell_batches) -> float:
    """Mean number of experts with nonzero gate per token over ``cell_batches`` — the Top-P k̄ efficiency
    lever (constant ``k`` for top-k arms). Hooks every ``MoEFFN``; returns ``nan`` for dense models."""
    moes = [m for m in model.modules() if isinstance(m, MoEFFN)]
    if not moes:
        return float("nan")
    tot = torch.zeros(())
    cnt = torch.zeros(())

    def hook(_mod, _inp, out):
        _y, g = out
        nz = (g > 0).sum(-1).float()                          # active experts per token
        tot.add_(nz.sum().cpu())
        cnt.add_(float(nz.numel()))

    handles = [m.register_forward_hook(hook) for m in moes]
    was_training = model.training
    model.eval()
    try:
        for cb in cell_batches:
            model(cb)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return float(tot / cnt) if float(cnt) > 0 else float("nan")
