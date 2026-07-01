"""End-to-end MoRE forward shapes for every routing arm (rel-f1 guarded)."""
from __future__ import annotations

import torch

from gloss.model.more import ROUTE_ONS, MoRE

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available


@rel_f1_available
def test_more_forward_all_route_ons():
    bundle, _task, cb = sample_cell_batch(seq_len=256, batch_size=8)
    name_emb = name_table()
    for route_on in ROUTE_ONS:
        model = MoRE(bundle, name_emb, d_model=64, d_sig=32, n_blocks=2, n_heads=4,
                     d_ff=128, enc_channels=64, route_on=route_on, num_experts=4, k=2)
        with torch.no_grad():
            logits, aux = model(cb)
        assert logits.shape == (cb.num_seeds, 1), route_on
        assert torch.isfinite(logits).all(), route_on
        assert torch.isfinite(aux) and float(aux) >= 0.0, route_on
        # dense arms have no router -> zero aux; MoE arms have a finite positive orthogonality loss
        if route_on in ("dense", "dense_wide"):
            assert float(aux) == 0.0, route_on
        else:
            assert float(aux) > 0.0, route_on


@rel_f1_available
def test_signature_arm_grads_reach_router_and_signature():
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    name_emb = name_table()
    model = MoRE(bundle, name_emb, d_model=64, d_sig=32, n_blocks=2, n_heads=4,
                 d_ff=128, enc_channels=64, route_on="signature", num_experts=4, k=2)
    logits, aux = model(cb)
    (logits.squeeze(-1).sum() + aux).backward()              # exercise task path + aux
    router_grad = model.substrate.blocks[0].ffn.router.weight.grad
    sig_grad = model.signature.schema_proj.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    assert sig_grad is not None and sig_grad.abs().sum() > 0


@rel_f1_available
def test_hybrid_arm_grads_reach_router_and_signature():
    """hybrid routes on [z ; h]: grads must reach BOTH the block router and the signature (the z half)."""
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    name_emb = name_table()
    model = MoRE(bundle, name_emb, d_model=64, d_sig=32, n_blocks=2, n_heads=4,
                 d_ff=128, enc_channels=64, route_on="hybrid", num_experts=4, k=2)
    logits, aux = model(cb)
    (logits.squeeze(-1).sum() + aux).backward()
    router_grad = model.substrate.blocks[0].ffn.router.weight.grad
    sig_grad = model.signature.schema_proj.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    assert sig_grad is not None and sig_grad.abs().sum() > 0


@rel_f1_available
def test_more_forward_with_architecture_additions():
    """Each S/C/P/H addition (and a combination) forwards finite on the signature arm."""
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    name_emb = name_table()
    configs = [
        dict(use_shared=True),                               # S
        dict(cosine=True, tau=0.3),                          # C
        dict(top_p=0.7),                                     # P
        dict(hmoe=True, n_groups=2, experts_per_group=2, k2=1),   # H
        dict(use_shared=True, cosine=True, top_p=0.7),       # S+C+P together (flat MoE)
    ]
    for extra in configs:
        model = MoRE(bundle, name_emb, d_model=64, d_sig=32, n_blocks=2, n_heads=4,
                     d_ff=128, enc_channels=64, route_on="signature", num_experts=4, k=2, **extra)
        with torch.no_grad():
            logits, aux = model(cb)
        assert logits.shape == (cb.num_seeds, 1), extra
        assert torch.isfinite(logits).all(), extra
        assert torch.isfinite(aux) and float(aux) >= 0.0, extra
