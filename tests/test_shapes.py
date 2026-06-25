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
