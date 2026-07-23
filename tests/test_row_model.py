"""R2 tests for RowModel and its submodules.

Hermetic sub-module tests use raw tensors / the chain fixture (no pytorch-frame stats). Full-model
tests need real stype stats, so they are rel-f1-guarded.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gloss.data.collate import column_vocab
from gloss.setjoin.collate import to_row_set_batch
from gloss.setjoin.model import _MoEStack
from gloss.setjoin.paths import row_paths
from gloss.setjoin.row_model import (
    CascadeCellPool,
    MeasurePoolStage,
    RowCellSignature,
    RowModel,
    RowSignature,
)

from ._join_fixtures import chain_bundle, make_chain_batch
from ._relf1 import name_table, sample_row_set_batch
from .conftest import rel_f1_available


# --------------------------------------------------------------------------------------------------
# hermetic: measure pool + cascade
# --------------------------------------------------------------------------------------------------
def test_measure_pool_stage_shape_grad_and_count_awareness():
    torch.manual_seed(0)
    st = MeasurePoolStage(d=16, n_query=4)
    H = torch.randn(5, 7, 16, requires_grad=True)
    valid = torch.ones(5, 7, dtype=torch.bool)
    out = st(H, valid)
    assert out.shape == (5, 4, 16)
    out.sum().backward()
    assert st.query.grad is not None and H.grad is not None
    # count-awareness: a row with 2 valid cells vs 7 valid cells -> different pooled vector
    v2 = torch.zeros(5, 7, dtype=torch.bool)
    v2[:, :2] = True
    a = MeasurePoolStage(d=16, n_query=1)
    torch.manual_seed(1)
    Hc = torch.randn(1, 7, 16)
    full = a(Hc, torch.ones(1, 7, dtype=torch.bool))
    part = a(Hc, torch.tensor([[True, True, False, False, False, False, False]]))
    assert not torch.allclose(full, part)


def test_cascade_cell_pool_reduces_to_one_vector():
    torch.manual_seed(0)
    pool = CascadeCellPool(d=16, widths=(8, 2, 1))
    H = torch.randn(6, 12, 16, requires_grad=True)
    valid = torch.ones(6, 12, dtype=torch.bool)
    valid[3:, 5:] = False
    out = pool(H, valid)
    assert out.shape == (6, 16) and torch.isfinite(out).all()
    out.sum().backward()
    assert H.grad is not None
    with pytest.raises(AssertionError):
        CascadeCellPool(d=16, widths=(8, 2))          # must end in width 1


# --------------------------------------------------------------------------------------------------
# hermetic: signatures (value-free by construction; test shape + path/hop sensitivity + grads)
# --------------------------------------------------------------------------------------------------
def _chain_rb(m_rows=8, cells_per_row=8):
    b = chain_bundle()
    rb = to_row_set_batch(make_chain_batch(), b, "order", m_rows=m_rows, cells_per_row=cells_per_row)
    return b, rb


def test_row_cell_signature_shape_path_sensitivity_and_grad():
    b, rb = _chain_rb()
    n_cols = len(column_vocab(b))
    name_emb = torch.randn(n_cols, 12)
    mod = torch.zeros(n_cols, dtype=torch.long)
    sig = RowCellSignature(name_emb, mod, n_stypes=1, n_paths=row_paths(b, "order").n_row_paths, d_sig=8)
    z = sig(rb)
    assert z.shape == (rb.num_seeds, rb.m_rows, rb.cells_per_row, 8) and torch.isfinite(z).all()
    # a cell's signature changes if only its path id changes (path term is live)
    rb_alt = replace(rb, cell_path_id=torch.zeros_like(rb.cell_path_id))
    assert not torch.allclose(sig(rb), sig(rb_alt))
    z.sum().backward()
    assert sig.schema_proj.weight.grad is not None and sig.path_emb.weight.grad is not None


def test_row_signature_shape_hop_sensitivity_and_row0_defined():
    b, rb = _chain_rb()
    sig = RowSignature(b.num_node_types, b.num_fk_roles, d_sig=8)
    z = sig(rb)
    assert z.shape == (rb.num_seeds, rb.m_rows, 8) and torch.isfinite(z).all()
    # Row 0 (hop 0, FK_NONE) has a defined, finite signature distinct from a child row (hop 1)
    assert torch.isfinite(z[0, 0]).all()
    assert not torch.allclose(z[0, 0], z[0, 1])


def test_dense_stack_has_zero_ortho_loss():
    st = _MoEStack(16, 2, 4, 0.0, route_on="dense", d_ff=32, d_route=16, num_experts=4, k=2)
    assert st.ortho_loss() == 0.0


# --------------------------------------------------------------------------------------------------
# rel-f1-guarded: full model
# --------------------------------------------------------------------------------------------------
def _small(bundle, name_emb, entity_table, **kw):
    return RowModel(bundle, name_emb, entity_table, d_model=32, n_cell_layers=1, n_row_layers=1,
                    n_heads=4, **kw)


@rel_f1_available
@pytest.mark.parametrize("route_on", ["signature", "hidden", "dense"])
def test_forward_all_route_arms(route_on):
    bundle, task, rb = sample_row_set_batch()
    name_emb = name_table()
    model = _small(bundle, name_emb, task.entity_table, route_on=route_on)
    logits, aux = model(rb)
    assert logits.shape == (rb.num_seeds, 1) and torch.isfinite(logits).all()
    assert torch.isfinite(aux)
    if route_on == "signature":
        assert float(aux) > 0
    if route_on == "dense":
        assert float(aux) == 0.0


@rel_f1_available
def test_param_count_under_30m():
    bundle, task, _rb = sample_row_set_batch()
    model = RowModel(bundle, name_table(), task.entity_table, d_model=256, n_cell_layers=2,
                     n_row_layers=2, d_ff=1024)
    assert sum(p.numel() for p in model.parameters()) < 30e6


@rel_f1_available
def test_grad_flow_to_router_signatures_and_pools():
    bundle, task, rb = sample_row_set_batch()
    model = _small(bundle, name_table(), task.entity_table, route_on="signature")
    logits, aux = model(rb)
    (logits.sum() + aux).backward()
    named = dict(model.named_parameters())
    for key in ["cell_enc.layers.0.ffn.router.weight", "row_enc.layers.0.ffn.router.weight",
                "cell_sig.schema_proj.weight", "row_sig.fk_emb.weight", "encoder.name_proj.weight"]:
        assert named[key].grad is not None, key
    # both pools get gradient
    assert any(p.grad is not None for n, p in named.items() if n.startswith("pool."))


@rel_f1_available
def test_empty_children_seed_is_finite():
    bundle, task, rb = sample_row_set_batch()
    # force every seed to have ONLY Row 0 (no children): drop rows 1..M
    rb2 = replace(rb, row_mask=torch.zeros_like(rb.row_mask))
    rb2.row_mask[:, 0] = True
    logits, aux = _small(bundle, name_table(), task.entity_table)(rb2)
    assert torch.isfinite(logits).all() and torch.isfinite(aux)


def _permute(rb, perm, axis):
    """Permute the M-row axis (axis='row') or the C-cell axis (axis='cell') and remap cell_placement."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    if axis == "row":
        idx = lambda t: t[:, perm]
        place = {nt: (b, inv[m], c, r, col) for nt, (b, m, c, r, col) in rb.cell_placement.items()}
        return replace(
            rb, cell_col_id=rb.cell_col_id[:, perm], cell_table_id=rb.cell_table_id[:, perm],
            cell_path_id=rb.cell_path_id[:, perm], cell_missing=rb.cell_missing[:, perm],
            cell_row_time=rb.cell_row_time[:, perm], cell_is_timed=rb.cell_is_timed[:, perm],
            cell_mask=rb.cell_mask[:, perm], row_mask=rb.row_mask[:, perm],
            row_table_id=rb.row_table_id[:, perm], row_fk_role=rb.row_fk_role[:, perm],
            row_hop=rb.row_hop[:, perm], row_row_time=rb.row_row_time[:, perm],
            row_is_timed=rb.row_is_timed[:, perm], cell_placement=place)
    else:
        place = {nt: (b, m, inv[c], r, col) for nt, (b, m, c, r, col) in rb.cell_placement.items()}
        return replace(
            rb, cell_col_id=rb.cell_col_id[:, :, perm], cell_table_id=rb.cell_table_id[:, :, perm],
            cell_path_id=rb.cell_path_id[:, :, perm], cell_missing=rb.cell_missing[:, :, perm],
            cell_row_time=rb.cell_row_time[:, :, perm], cell_is_timed=rb.cell_is_timed[:, :, perm],
            cell_mask=rb.cell_mask[:, :, perm], cell_placement=place)


@rel_f1_available
@pytest.mark.parametrize("axis", ["row", "cell"])
def test_permutation_invariance(axis):
    bundle, task, rb = sample_row_set_batch()
    model = _small(bundle, name_table(), task.entity_table, aggregate="mean").eval()
    with torch.no_grad():
        base, _ = model(rb)
        n = rb.m_rows if axis == "row" else rb.cells_per_row
        perm = torch.randperm(n)
        perm_out, _ = model(_permute(rb, perm, axis))
    assert torch.allclose(base, perm_out, atol=1e-4)


@rel_f1_available
def test_gradient_checkpointing_cells_runs():
    bundle, task, rb = sample_row_set_batch(batch_size=6)
    model = _small(bundle, name_table(), task.entity_table, route_on="signature",
                   checkpoint_cells=True).train()
    logits, aux = model(rb)
    (logits.sum() + aux).backward()
    assert torch.isfinite(logits).all()
    assert model.cell_enc.layers[0].ffn.router.weight.grad is not None


@rel_f1_available
def test_overfit_one_batch_binary():
    bundle, task, rb = sample_row_set_batch(batch_size=8)
    model = _small(bundle, name_table(), task.entity_table, route_on="signature")
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    y, m = rb.target, rb.has_target
    for _ in range(150):
        opt.zero_grad()
        logits, aux = model(rb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1)[m], y[m]) + 0.01 * aux
        loss.backward()
        opt.step()
    assert float(loss) < 0.35


# --------------------------------------------------------------------------------------------------
# R3: aggregate / counts / row-pool / cell-slots arms
# --------------------------------------------------------------------------------------------------
@rel_f1_available
@pytest.mark.parametrize("kw", [
    {"aggregate": "slot"},
    {"aggregate": "slot", "agg_slots": 2},
    {"use_counts": True},
    {"aggregate": "slot", "use_counts": True},
    {"row_pool": "gated"},
    {"cell_slots": (4,)},
    {"cell_slots": (16, 4, 2)},
])
def test_aggregate_and_pool_arms_forward_and_grad(kw):
    bundle, task, rb = sample_row_set_batch()
    model = _small(bundle, name_table(), task.entity_table, route_on="signature", **kw)
    logits, aux = model(rb)
    assert logits.shape == (rb.num_seeds, 1) and torch.isfinite(logits).all()
    (logits.sum() + aux).backward()
    if kw.get("aggregate") == "slot":
        assert any(p.grad is not None for n, p in model.named_parameters() if n.startswith("agg_pool."))
    if kw.get("use_counts"):
        assert model.w_cnt is not None and model.w_cnt.weight.grad is not None


@rel_f1_available
def test_use_counts_widens_head_and_regression_out_dim():
    bundle, task, rb = sample_row_set_batch()
    plain = _small(bundle, name_table(), task.entity_table)
    counted = _small(bundle, name_table(), task.entity_table, use_counts=True)
    assert plain.head[0].normalized_shape == (32,)
    assert counted.head[0].normalized_shape == (64,)              # concatenated log-counts -> 2d
    reg = _small(bundle, name_table(), task.entity_table, out_dim=1, aggregate="slot")
    assert reg(rb)[0].shape == (rb.num_seeds, 1)
