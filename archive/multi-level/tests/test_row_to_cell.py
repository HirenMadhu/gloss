"""`RowToCellAttention` — §3.6 as attention instead of an additive own-row broadcast.

The claim this operator makes is narrow and testable: *a cell can weight a particular FK-adjacent
row*. Everything `Broadcast` already did — neighbour information reaching cells at all — was true
before, via row attention followed by the own-row add. So the tests that matter are the two that
separate the mechanisms:

* :func:`test_two_cells_in_one_row_can_read_the_row_graph_differently` — under `Broadcast` every cell
  of a row receives an identical update; here they need not.
* :func:`test_a_cell_cannot_see_a_non_adjacent_row` — the FK structure is a real constraint, not
  decoration. Without it this is just full attention over rows and the role bias is meaningless.
"""
from __future__ import annotations

import pytest
import torch

from gloss.model.row_level import Broadcast, RowToCellAttention
from gloss.model.time_encoding import TimeLadder
from gloss.model.two_level import BROADCAST_MODES, TwoLevelBlock

from .test_row_level import D_MODEL, D_SIG, D_TEXT, _tables, stub_batch

N_HEADS = 4


def build(K, **kw):
    _, role, _ = _tables(K)
    torch.manual_seed(0)
    return RowToCellAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS, **kw)


def inputs(cb, R, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(cb.num_seeds, cb.seq_len, D_MODEL),
            torch.randn(cb.num_seeds, R, D_MODEL))


# ------------------------------------------------------------------ contract ---
def test_shapes_and_finiteness():
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    out, diag = build(K)(h, u, cb)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    assert {"r2c_entropy", "r2c_own_row_mass", "r2c_gamma_abs_mean"} <= set(diag)
    assert 0.0 <= float(diag["r2c_own_row_mass"]) <= 1.0


def test_a_cell_cannot_see_a_non_adjacent_row():
    """The FK graph must actually constrain who a cell reads.

    In the fixture row 1's neighbours are itself and the root (row 0); row 2 is not adjacent to it.
    Perturbing row 2's token must leave row 1's cells untouched, while perturbing row 0's must not.
    """
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    op = build(K)
    base, _ = op(h, u, cb)

    cells_of_row_1 = (cb.cell_row[0] == 1).nonzero().flatten()
    assert len(cells_of_row_1), "fixture no longer puts any cell in row 1"
    assert int(cb.adj_role[0, 1, 2]) == 0, "fixture changed: row 2 is now adjacent to row 1"

    far = u.clone(); far[0, 2] += 5.0
    out_far, _ = op(h, far, cb)
    assert torch.allclose(base[0, cells_of_row_1], out_far[0, cells_of_row_1], atol=1e-6), \
        "a cell read a row its own row has no FK edge to"

    near = u.clone(); near[0, 0] += 5.0
    out_near, _ = op(h, near, cb)
    assert not torch.allclose(base[0, cells_of_row_1], out_near[0, cells_of_row_1], atol=1e-4), \
        "a cell ignored its adjacent (parent) row"


def test_two_cells_in_one_row_can_read_the_row_graph_differently():
    """The whole point of the change, stated as a contrast with what it replaces."""
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    pair = (cb.cell_row[0] == 1).nonzero().flatten()[:2]
    assert len(pair) == 2, "fixture no longer has two cells sharing a row"

    torch.manual_seed(0)
    bc = Broadcast(D_MODEL, mode="additive")
    bc_delta = bc(h, u, cb) - h
    assert torch.allclose(bc_delta[0, pair[0]], bc_delta[0, pair[1]], atol=1e-6), \
        "Broadcast is supposed to give both cells the identical vector; the contrast is void"

    out, _ = build(K)(h, u, cb)
    delta = out - h
    assert not torch.allclose(delta[0, pair[0]], delta[0, pair[1]], atol=1e-4), \
        "both cells of a row got the same update — this has collapsed back to Broadcast"


def test_the_self_loop_means_a_cell_never_loses_its_own_row():
    """A row with no FK neighbours must still be readable by its own cells, or its cells go blind."""
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    adj = cb.adj_role.clone()
    adj[:] = 0
    for r in range(R):                       # self-loops only
        adj[:, r, r] = 2 * K + 1
    cb.adj_role = adj
    h, u = inputs(cb, R)
    out, diag = build(K)(h, u, cb)
    assert torch.isfinite(out).all()
    assert float(diag["r2c_own_row_mass"]) == pytest.approx(1.0, abs=1e-5)


def test_padding_cells_pass_through_untouched():
    """A pad slot has no row, so it must get exactly its residual — not row 0's content."""
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    out, _ = build(K)(h, u, cb)
    pad = cb.is_padding
    assert torch.isfinite(out).all()
    assert torch.allclose(out[pad], h[pad], atol=1e-6)
    assert pad.any(), "fixture no longer exercises padding"


def test_cells_whose_row_was_truncated_away_are_not_misrouted():
    """`cell_row >= R` (a row that MAX_ROWS cut) must be dropped, not clamped onto the last row."""
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    cb.cell_row = cb.cell_row.clone()
    cb.cell_row[0, 0] = R + 3
    h, u = inputs(cb, R)
    out, _ = build(K)(h, u, cb)
    assert torch.allclose(out[0, 0], h[0, 0], atol=1e-6)


@pytest.mark.parametrize("role_bias,time_bias", [("name_derived", "rope"), ("none", "none"),
                                                 ("name_derived", "none"), ("none", "rope")])
def test_every_bias_combination_runs(role_bias, time_bias):
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    out, diag = build(K, role_bias=role_bias, time_bias=time_bias)(h, u, cb)
    assert torch.isfinite(out).all()
    assert ("r2c_gamma_abs_mean" in diag) == (role_bias == "name_derived")


def test_gradients_reach_the_role_bias_parameters():
    """γ is the only thing that lets a cell tell a parent from a child. If it gets no gradient the
    operator degenerates to content-only attention over an unlabelled neighbourhood."""
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    h, u = inputs(cb, R)
    op = build(K)
    out, _ = op(h, u, cb)
    out.square().sum().backward()
    for name in ("w_rho.weight", "v_head", "c_dir", "w_q.weight", "w_k.weight", "w_v.weight"):
        p = dict(op.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} got a zero gradient"


def test_no_dataset_sized_entry_lands_in_the_state_dict():
    """§0: the role table is frozen and non-persistent, so a checkpoint stays loadable on a schema
    with a different K. A K-shaped state_dict row count would make that unrecoverable."""
    # K=5 so that K+1=6 collides with none of n_heads(4) / N_DIR(3) / d_sig(16) / d_model(32) —
    # the default K=3 makes K+1 equal n_heads and the check passes or fails for the wrong reason
    cb, K = stub_batch(K=5)
    op = build(K)
    K1 = op.role_name_emb.shape[0]
    assert K1 not in (N_HEADS, D_MODEL, D_SIG, 3), "fixture no longer isolates the role axis"
    for name, t in op.state_dict().items():
        assert K1 not in tuple(t.shape), f"{name} has a role-count-shaped axis {tuple(t.shape)}"


# -------------------------------------------------------------------- wiring ---
def test_the_block_merges_the_diag_and_leaves_other_modes_alone():
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    tab, role, col = _tables(K)
    h, u = inputs(cb, R)
    torch.manual_seed(0)
    z = torch.randn(cb.num_seeds, cb.seq_len, D_SIG)
    s = torch.randn(cb.num_seeds, R, D_SIG)

    def run(mode):
        torch.manual_seed(0)
        blk = TwoLevelBlock(D_MODEL, 2 * D_MODEL, D_SIG, TimeLadder(), col, role,
                            n_heads=N_HEADS, cell_attention="full", broadcast=mode)
        return blk(h, u, z, s, cb)

    _, _, _, _, d_attn = run("attention")
    _, _, _, _, d_add = run("additive")
    assert "r2c_own_row_mass" in d_attn
    assert not any(k.startswith("r2c_") for k in d_add)


def test_attention_is_not_the_default_and_unknown_modes_are_rejected():
    """Every result to date used `additive`; the mode must not switch itself on."""
    assert BROADCAST_MODES[0] == "additive"
    cb, K = stub_batch()
    tab, role, col = _tables(K)
    assert isinstance(TwoLevelBlock(D_MODEL, 2 * D_MODEL, D_SIG, TimeLadder(), col, role,
                                    n_heads=N_HEADS).broadcast, Broadcast)
    with pytest.raises(ValueError, match="broadcast"):
        TwoLevelBlock(D_MODEL, 2 * D_MODEL, D_SIG, TimeLadder(), col, role,
                      n_heads=N_HEADS, broadcast="nope")


def test_scores_stay_at_cell_by_ROW_not_cell_by_cell():
    """Cost guard. The mask and γ depend only on (ν(i), s), so they are built at row×row and gathered
    — a future edit that expands them to cell resolution before use would multiply the memory by
    R and pass every numerical test above."""
    from unittest.mock import patch

    cb, K = stub_batch(B=2, R=6, S=24)
    R, S = cb.adj_role.shape[1], cb.seq_len
    h, u = inputs(cb, R)
    op = build(K)
    seen: list[tuple] = []

    real_expand = torch.Tensor.expand

    def record(fn):
        def wrapper(self, *a, **kw):
            out = fn(self, *a, **kw)
            seen.append(tuple(out.shape))
            return out
        return wrapper

    with patch.object(torch.Tensor, "expand", record(real_expand)):
        op(h, u, cb)
    assert seen, "instrumentation caught nothing"
    B, H = cb.num_seeds, N_HEADS
    # per-cell tensors are fine (theta is [B,S,n_freq]); cell-BY-CELL is not, and nothing may exceed
    # the cell-by-row score tensor, which is the intended ceiling for this operator
    assert not any(sh.count(S) >= 2 for sh in seen), "a cell-by-cell tensor was built"
    budget = B * H * S * R
    biggest = max(int(torch.tensor(sh).prod()) for sh in seen)
    assert biggest <= budget, f"{biggest} exceeds the cell-by-row budget {budget}"
