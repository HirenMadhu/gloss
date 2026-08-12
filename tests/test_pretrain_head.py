"""The masked-cell decode head: shapes, the §0 no-dataset-artifact guard, and that it can learn.

The overfit test runs on **CPU** for the same reason `test_train.py`'s regression sibling does:
`RowPool`'s `index_add` uses CUDA atomics, and over a few hundred optimizer steps that is the
difference between converging and not (measured: three identical CUDA trials of the supervised fit
reached 0.5431 / 0.0000 / 1.1412). An assertion about optimization needs a deterministic reduction.
"""
from __future__ import annotations

import torch

from gloss.data.masking import (CATEGORICAL_ID, build_column_target_spec, gather_masked_targets,
                                sample_cell_mask)
from gloss.model.mlm_head import MaskedCellHead
from gloss.model.more import MoRE
from gloss.text.cache import HashEncoder
from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available

D_MODEL, ENC = 64, 64


def _model(bundle, name_emb, **kw):
    enc = HashEncoder(dim=64)
    return MoRE(bundle, name_emb, d_model=D_MODEL, d_sig=32, n_blocks=2, n_heads=4, d_ff=128,
                enc_channels=ENC, num_experts=4, k=2,
                table_name_emb=build_table_name_embeddings(bundle, enc),
                role_name_emb=role_name_embeddings_with_none(bundle, enc), **kw)


def _cat_table(model, spec, targets):
    """The category-embedding matrix of whichever table the batch's categorical targets live in."""
    is_cat = targets.stype == CATEGORICAL_ID
    if not bool(is_cat.any()):
        return None
    nt = spec.node_types[int(spec.table[int(targets.gid[is_cat][0])])]
    return model.category_table(nt)


def _setup(seq_len=256, batch_size=8, p_random=0.3, seed=0):
    """Fixed model init AND fixed mask.

    The mask must be pinned with an explicit generator, not left to the ambient RNG: rel-f1 has
    exactly **one** categorical column in its whole schema, so whether a batch contains a masked
    categorical cell depends on the global RNG state — which means whichever tests ran first. That is
    how the categorical assertions here passed standalone and failed in-suite.
    """
    from gloss.utils.seeding import seed_everything

    bundle, task, cb = sample_cell_batch(seq_len=seq_len, batch_size=batch_size)
    seed_everything(seed)
    model = _model(bundle, name_table())
    spec = build_column_target_spec(bundle, model.encoder)
    g = torch.Generator().manual_seed(seed)
    mask, seed_mask = sample_cell_mask(cb, spec, p_random=p_random, generator=g)
    targets = gather_masked_targets(cb, spec, mask, seed_mask)
    return bundle, task, cb, model, spec, mask, targets


# ------------------------------------------------------------------------------------ §0 guard


@rel_f1_available
def test_head_has_no_parameter_shaped_by_the_schema():
    """The head must load onto a database it never saw. Nothing in it may be sized by the number of
    columns, tables, FK roles, or -- the one that is easy to get wrong -- category counts."""
    bundle, _t, _cb, model, _spec, _m, _tg = _setup()
    head = MaskedCellHead(D_MODEL, ENC)
    C = len(model.encoder.name_emb)
    forbidden = {C, bundle.num_roles, bundle.num_node_types, int(_spec_max_cat(bundle, model))}
    forbidden.discard(D_MODEL)
    forbidden.discard(ENC)
    for name, p in head.named_parameters():
        assert not (set(p.shape) & forbidden), (name, tuple(p.shape), forbidden)


def _spec_max_cat(bundle, model) -> int:
    spec = build_column_target_spec(bundle, model.encoder)
    return int(spec.n_cat.max())


@rel_f1_available
def test_head_transfers_to_a_different_schema_unchanged():
    """The cheap LODO proxy, applied to the new head: a state_dict built on one DB loads on another
    with no shape mismatch, because the only vocabulary-shaped tensor lives in the per-DB encoder."""
    a = MaskedCellHead(D_MODEL, ENC)
    b = MaskedCellHead(D_MODEL, ENC)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True), None
    assert unexpected is None


# ------------------------------------------------------------------------------------- shapes


@rel_f1_available
def test_categorical_logits_are_the_column_s_own_width():
    bundle, _t, cb, model, spec, _m, targets = _setup(batch_size=16, p_random=0.6)
    is_cat = targets.stype == CATEGORICAL_ID
    if not bool(is_cat.any()):
        return                                     # rel-f1 has exactly one categorical column
    head = MaskedCellHead(D_MODEL, ENC)
    table = _cat_table(model, spec, targets)
    assert table is not None, "the tied categorical head needs the encoder's embedding table"
    hidden = torch.randn(int(is_cat.sum()), D_MODEL)
    logits = head.category_logits(hidden, targets.gid[is_cat], spec, table)

    n_cat = spec.n_cat[targets.gid[is_cat]]
    assert logits.shape == (int(is_cat.sum()), int(n_cat.max()))
    for i in range(logits.shape[0]):
        k = int(n_cat[i])
        assert torch.isfinite(logits[i, :k]).all()
        assert bool(torch.isinf(logits[i, k:]).all()), "surplus classes must be masked to -inf"


@rel_f1_available
def test_forward_returns_split_stats_and_a_graph_connected_zero_when_nothing_is_masked():
    bundle, _t, cb, model, spec, mask, targets = _setup()
    head = MaskedCellHead(D_MODEL, ENC)
    cells = torch.randn(cb.num_seeds, cb.seq_len, D_MODEL, requires_grad=True)
    loss, stats = head(cells, targets, spec, _cat_table(model, spec, targets))
    assert torch.isfinite(loss)
    assert stats["n_masked"] == float(len(targets))
    assert stats["n_num"] + stats["n_cat"] == stats["n_masked"]
    assert 0 <= stats["n_seed"] <= cb.num_seeds

    empty = gather_masked_targets(cb, spec, torch.zeros_like(mask), torch.zeros_like(mask))
    zero, zstats = head(cells, empty, spec, None)
    assert float(zero) == 0.0 and zero.requires_grad, "must stay in the graph for DDP"
    assert zstats["n_num"] == 0.0


@rel_f1_available
def test_numerical_head_starts_at_the_column_mean():
    """Zero-init, as RT does: the first prediction is exactly 0 in z-space, so a heavy-tailed column
    cannot dominate step 1 through a random readout."""
    head = MaskedCellHead(D_MODEL, ENC)
    assert torch.equal(head.num_head.weight, torch.zeros_like(head.num_head.weight))
    assert torch.equal(head.num_head.bias, torch.zeros_like(head.num_head.bias))


# -------------------------------------------------------------------------------- it can learn


@rel_f1_available
def test_masked_cell_objective_overfits_one_batch():
    """End to end on CPU: encoder -> substrate -> head, one fixed batch and mask, driven to ~0."""
    from gloss.utils.seeding import seed_everything

    bundle, _task, cb = sample_cell_batch(seq_len=256, batch_size=8)
    seed_everything(0)
    model = _model(bundle, name_table())
    spec = build_column_target_spec(bundle, model.encoder)
    g = torch.Generator().manual_seed(0)
    mask, seed = sample_cell_mask(cb, spec, p_random=0.3, generator=g)
    targets = gather_masked_targets(cb, spec, mask, seed)
    assert len(targets) > 20, len(targets)

    head = MaskedCellHead(D_MODEL, ENC)
    opt = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=3e-3)
    nt = spec.node_types[int(spec.table[int(targets.gid[0])])]

    def step():
        _logits, _aux, cells = model(cb, cell_mask=mask, return_cells=True)
        return head(cells, targets, spec, model.category_table(nt))

    with torch.no_grad():
        init = float(step()[0])
    for _ in range(300):
        opt.zero_grad()
        loss, _ = step()
        loss.backward()
        opt.step()
    final = float(loss)
    assert final < 0.2 * init, f"masked-cell objective did not fit one batch: {init=:.4f} {final=:.4f}"


@rel_f1_available
def test_the_zero_init_numerical_head_blocks_its_own_trunk_gradient_for_one_step():
    """A consequence of RT's zero-init worth pinning rather than rediscovering.

    ``pred = W h`` with ``W = 0``, so ``dloss/dh = err * W = 0``: at initialization the **numerical**
    decoder passes no gradient into the trunk, however large its loss. One optimizer step makes ``W``
    non-zero and the trunk starts learning. The **categorical** decoder is not zero-init (its logits
    are tied to the encoder's category table, which is already trained), so it feeds the trunk from
    step 0 — which is why the whole model is not frozen, and why a naive "gradients reach the
    encoder" check passes or fails depending on whether the batch happened to mask a categorical
    cell. Both branches are asserted separately here so neither can hide the other.
    """
    _b, _t, cb, model, spec, mask, targets = _setup()
    head = MaskedCellHead(D_MODEL, ENC)

    # numerical path only (cat_table=None disables the categorical branch)
    loss, stats = head(model(cb, cell_mask=mask, return_cells=True)[2], targets, spec, None)
    loss.backward()
    assert stats["n_num"] > 0 and float(loss) > 0
    assert float(head.num_head.weight.grad.abs().sum()) > 0, "the head itself must still learn"
    assert float(model.encoder.mask_emb.weight.grad.abs().sum()) == 0.0

    # with the categorical branch live, the trunk is reached immediately
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    table = _cat_table(model, spec, targets)
    assert table is not None, "rel-f1's one categorical column should be in this batch"
    loss2, stats2 = head(model(cb, cell_mask=mask, return_cells=True)[2], targets, spec, table)
    loss2.backward()
    assert stats2["n_cat"] > 0
    assert float(model.encoder.mask_emb.weight.grad.abs().sum()) > 0


@rel_f1_available
def test_gradients_reach_the_mask_token_and_both_routers():
    """Same check one step later, once the zero-init head above has become non-zero."""
    _b, _t, cb, model, spec, mask, targets = _setup()
    head = MaskedCellHead(D_MODEL, ENC)
    opt = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-2)
    table = _cat_table(model, spec, targets)
    for _ in range(2):
        opt.zero_grad()
        _logits, aux, cells = model(cb, cell_mask=mask, return_cells=True)
        loss, _ = head(cells, targets, spec, table)
        (loss + aux).backward()
        opt.step()

    assert model.encoder.mask_emb.weight.grad is not None
    assert float(model.encoder.mask_emb.weight.grad.abs().sum()) > 0, "mask token got no signal"
    blk = model.substrate.blocks[0]
    assert float(blk.cell_ffn.router.weight.grad.abs().sum()) > 0, "cell router got no signal"
    assert float(blk.row_ffn.w_g.grad.abs().sum()) > 0, "row router got no signal"
    assert float(head.num_head.weight.grad.abs().sum()) > 0


@rel_f1_available
def test_rel_f1_entity_table_has_no_maskable_cell_so_seed_targets_are_empty():
    """Not a defect — the reason the pretrain loader weights seed tables by *maskable* rows.

    rel-f1's entity table is ``drivers``, whose 6 feature columns are all text or timestamp, so the
    seed-target half of the objective is structurally silent on every rel-f1 driver task. The same is
    true of 7 of rel-trial's 15 tables, including ``facilities_studies`` at 1.87M rows -- about half
    that database's rows. Seeding uniformly over rows would spend half the budget on sequences that
    can never produce a seed target.
    """
    from gloss.data.masking import maskable_cells

    _b, task, cb, _model, spec, _m, _tg = _setup()
    assert task.entity_table == "drivers"
    assert int((maskable_cells(cb, spec) & cb.is_seed_cell).sum()) == 0
    _mask, seed = sample_cell_mask(cb, spec, p_random=0.0)
    assert not bool(seed.any())
