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
    router, at recency bin 0 (signature arm only; ``{}`` otherwise). Cluster columns by value to see
    cross-table semantic groups."""
    sig = getattr(model, "signature", None)
    if sig is None:
        return {}
    router = next((m.router for m in model.substrate.modules() if isinstance(m, MoEFFN)), None)
    if router is None:
        return {}
    C = int(sig.name_emb.shape[0])
    rec = torch.zeros(C, dtype=torch.long, device=sig.name_emb.device)
    z = sig.norm(sig.schema_proj(sig.name_emb) + sig.stype_emb(sig.modality_id) + sig.recency_emb(rec))
    expert = router(z).argmax(dim=-1)                         # [C]
    return {i: int(expert[i]) for i in range(C)}
