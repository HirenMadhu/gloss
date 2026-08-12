"""End-to-end MoRE forward shapes for both surviving routing arms (rel-f1 guarded).

The S/C/P/H MoE additions and the `hybrid`/`hidden`/`value`/`identity`/`dense_wide` arms are retired
to ``archive/multi-level/``; what remains is the headline comparison, `signature` vs `dense`.
"""
from __future__ import annotations

import torch

from gloss.model.more import ROUTE_ONS, MoRE
from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available


def _tables(bundle):
    from gloss.text.cache import HashEncoder

    enc = HashEncoder(dim=64)
    return (build_table_name_embeddings(bundle, enc),
            role_name_embeddings_with_none(bundle, enc))


def _model(bundle, name_emb, **kw):
    tab, role = _tables(bundle)
    return MoRE(bundle, name_emb, d_model=64, d_sig=32, n_blocks=2, n_heads=4,
                d_ff=128, enc_channels=64, num_experts=4, k=2,
                table_name_emb=tab, role_name_emb=role, **kw)


@rel_f1_available
def test_more_forward_all_route_ons():
    bundle, _task, cb = sample_cell_batch(seq_len=256, batch_size=8)
    name_emb = name_table()
    assert set(ROUTE_ONS) == {"signature", "dense"}, "the arm set changed; update this test"
    for route_on in ROUTE_ONS:
        model = _model(bundle, name_emb, route_on=route_on)
        with torch.no_grad():
            logits, aux = model(cb)
        assert logits.shape == (cb.num_seeds, 1), route_on
        assert torch.isfinite(logits).all(), route_on
        assert torch.isfinite(aux) and float(aux) >= 0.0, route_on
        # `dense` drops the CELL router, but the ROW MoE is always on, so aux stays positive at both
        # arms. What separates them is where the aux comes from — checked below, per level.
        assert float(aux) > 0.0, route_on


@rel_f1_available
def test_dense_arm_has_no_cell_router_but_keeps_the_row_one():
    """The arm difference is structural, not just numeric: `dense` must remove the cell-level MoE
    entirely, or `signature` vs `dense` is not measuring the mechanism it claims to."""
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    name_emb = name_table()
    sig = _model(bundle, name_emb, route_on="signature")
    dense = _model(bundle, name_emb, route_on="dense")

    assert hasattr(sig.substrate.blocks[0].cell_ffn, "router")
    assert not hasattr(dense.substrate.blocks[0].cell_ffn, "router")
    assert hasattr(dense.substrate.blocks[0].row_ffn, "w_g"), "the ROW MoE must survive `dense`"

    with torch.no_grad():
        _c, _r, _aux, diag = dense.substrate(dense.encoder(cb), cb, z=dense.signature(cb))
    assert float(diag["aux_cell"]) == 0.0, "dense arm still produced a cell-level aux term"
    assert float(diag["aux_row"]) > 0.0


@rel_f1_available
def test_signature_arm_grads_reach_router_and_signature():
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    model = _model(bundle, name_table(), route_on="signature")
    logits, aux = model(cb)
    (logits.squeeze(-1).sum() + aux).backward()              # exercise task path + aux
    router_grad = model.substrate.blocks[0].cell_ffn.router.weight.grad
    sig_grad = model.signature.schema_proj.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    assert sig_grad is not None and sig_grad.abs().sum() > 0


@rel_f1_available
def test_grads_reach_both_levels_routers():
    """The MoE is at BOTH levels; a gradient that reaches only one means a level is inert."""
    bundle, _task, cb = sample_cell_batch(seq_len=128, batch_size=8)
    model = _model(bundle, name_table(), route_on="signature")
    logits, aux = model(cb)
    (logits.squeeze(-1).sum() + aux).backward()
    for name, g in (("cell", model.substrate.blocks[0].cell_ffn.router.weight.grad),
                    ("row", model.substrate.blocks[0].row_ffn.w_g.grad)):
        assert g is not None and g.abs().sum() > 0, f"{name} router got no gradient"
