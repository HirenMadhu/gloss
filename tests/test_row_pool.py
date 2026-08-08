"""`RowPool` as a segment softmax — equivalence against the dense `[B,R,M,S]` reference.

The rewrite is a pure efficiency change, so the bar is *exactness*, not "close enough": the dense
reference below is the code that produced every result so far, and any disagreement means the row
level now computes something different. Each cell belongs to exactly one row, so `1/R` of a dense
score tensor survives its mask — 99.4% waste at the shapes we train on — but the surviving softmax
is over exactly the same sets, which is what these tests pin.
"""
from __future__ import annotations

import pytest
import torch

from gloss.model.row_level import RowPool

MODES = ("mean", "signature", "hidden", "hybrid")


def dense_reference(pool: RowPool, h, u, s, cb):
    """The pre-rewrite implementation, verbatim, as the oracle."""
    B, R, _ = u.shape
    member = pool._membership(cb, R)                                  # [B,R,S]
    if pool.mode == "mean":
        cnt = member.sum(-1, keepdim=True).clamp_min(1)
        return u + (member.to(h.dtype) @ h) / cnt
    q_in = {"signature": s, "hidden": u, "hybrid": torch.cat([u, s], dim=-1)}[pool.mode]
    q = pool.w_q(q_in).view(B, R, pool.slots, pool.d_h)
    k = pool.w_k(pool.col_name_emb[cb.col_idxs.clamp_min(0)])
    v = pool.w_v(h)
    scores = torch.einsum("brmd,bsd->brms", q, k) / pool.d_h ** 0.5
    scores = scores.masked_fill(~member.unsqueeze(2), float("-inf"))
    empty = ~member.any(-1)
    a = torch.softmax(scores, dim=-1)
    a = torch.where(empty.view(B, R, 1, 1), torch.zeros_like(a), a)
    pooled = torch.einsum("brms,bsd->brmd", a, v).reshape(B, R, pool.slots * pool.d_h)
    return u + pool.w_o(pooled)


class _CB:
    """Minimal stand-in for the `CellBatch` fields `RowPool` reads."""

    def __init__(self, cell_row, is_padding, col_idxs):
        self.cell_row = cell_row
        self.is_padding = is_padding
        self.col_idxs = col_idxs


def make_case(B=3, R=6, S=17, C=9, d=16, d_sig=8, d_text=12, *, seed=0,
              empty_rows=True, truncated=False, all_padding_seed=False):
    """A batch with the awkward cases baked in: empty rows, padding, and (optionally) cells whose
    row index was truncated away by `MAX_ROWS` (`row >= R`), which the dense path dropped silently."""
    g = torch.Generator().manual_seed(seed)
    hi = R + 3 if truncated else R
    cell_row = torch.randint(0, hi, (B, S), generator=g)
    is_padding = torch.rand(B, S, generator=g) < 0.25
    if empty_rows:
        cell_row[cell_row == R - 1] = 0           # nothing maps to the last row slot
    if all_padding_seed:
        is_padding[0] = True                      # a seed whose every cell is padding
    cell_row[is_padding] = -1                     # collate's pad sentinel
    col_idxs = torch.randint(-1, C, (B, S), generator=g)
    cb = _CB(cell_row, is_padding, col_idxs)
    h = torch.randn(B, S, d, generator=g)
    u = torch.randn(B, R, d, generator=g)
    s = torch.randn(B, R, d_sig, generator=g)
    name_emb = torch.randn(C, d_text, generator=g)
    return cb, h, u, s, name_emb


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["plain", "truncated", "all_padding_seed"])
def test_segment_softmax_matches_the_dense_reference(mode, case):
    cb, h, u, s, name_emb = make_case(truncated=(case == "truncated"),
                                      all_padding_seed=(case == "all_padding_seed"))
    torch.manual_seed(0)
    pool = RowPool(h.shape[-1], s.shape[-1], name_emb, slots=4, mode=mode)
    got = pool(h, u, s, cb)
    want = dense_reference(pool, h, u, s, cb)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got - want).abs().max()


@pytest.mark.parametrize("mode", [m for m in MODES if m != "mean"])
def test_gradients_match_the_dense_reference(mode):
    """Equal outputs with unequal gradients would train differently while testing identically."""
    cb, h, u, s, name_emb = make_case()
    torch.manual_seed(0)
    pool = RowPool(h.shape[-1], s.shape[-1], name_emb, slots=4, mode=mode)

    def grads(fn):
        pool.zero_grad()
        hh = h.clone().requires_grad_(True)
        fn(pool, hh, u, s, cb).square().sum().backward()
        return hh.grad, {n: p.grad.clone() for n, p in pool.named_parameters() if p.grad is not None}

    g_new, p_new = grads(lambda p, *a: p(*a))
    g_ref, p_ref = grads(dense_reference)
    assert torch.allclose(g_new, g_ref, atol=1e-5, rtol=1e-5), (g_new - g_ref).abs().max()
    assert set(p_new) == set(p_ref)
    for n in p_new:
        assert torch.allclose(p_new[n], p_ref[n], atol=1e-5, rtol=1e-5), n


def test_rows_with_no_cells_pool_to_exactly_zero():
    """An empty row must contribute nothing, not NaN — `12.9%` of rel-event's rows are empty at
    `seq_len=512`, so this is the common case there, not an edge case."""
    cb, h, u, s, name_emb = make_case()
    torch.manual_seed(0)
    pool = RowPool(h.shape[-1], s.shape[-1], name_emb, slots=4, mode="hybrid")
    out = pool(h, u, s, cb)
    R = u.shape[1]
    occupied = torch.zeros(u.shape[0], R, dtype=torch.bool)
    for b in range(u.shape[0]):
        rows = cb.cell_row[b][~cb.is_padding[b]]
        occupied[b, rows[(rows >= 0) & (rows < R)]] = True
    assert torch.isfinite(out).all()
    # empty rows pass through the residual untouched
    assert torch.allclose(out[~occupied], u[~occupied], atol=1e-6)
    assert (~occupied).any(), "fixture no longer exercises empty rows"


def test_it_never_materializes_a_dense_row_by_cell_tensor():
    """The point of the rewrite. A future edit that reintroduces `[B,R,M,S]` (or even `[B,R,S]`)
    would pass every numerical test above while undoing the ~160x memory saving, so assert on the
    shapes actually allocated."""
    from unittest.mock import patch

    cb, h, u, s, name_emb = make_case(B=2, R=8, S=20)
    torch.manual_seed(0)
    pool = RowPool(h.shape[-1], s.shape[-1], name_emb, slots=4, mode="hybrid")
    B, R, S, M = u.shape[0], u.shape[1], h.shape[1], pool.slots

    seen: list[tuple] = []

    def record(fn):
        def wrapper(*a, **kw):
            out = fn(*a, **kw)
            seen.append(tuple(out.shape))
            return out
        return wrapper

    with patch.object(torch.Tensor, "expand", record(torch.Tensor.expand)), \
         patch.object(torch.Tensor, "gather", record(torch.Tensor.gather)):
        pool(h, u, s, cb)
    assert seen, "instrumentation caught nothing — the spy no longer covers the hot path"

    # The invariant: nothing may scale with R*S. `q_cell` is [B,S,M,dh], which grows with M*dh and
    # not with R, so a size bound alone would not catch a regression — the joint R-and-S check is
    # what actually pins it.
    for shape in seen:
        assert not (R in shape and S in shape), f"dense row-by-cell tensor {shape} reintroduced"
    dense_budget = B * R * M * S            # what the pre-rewrite score tensor cost
    assert max(int(torch.tensor(sh).prod()) for sh in seen) < dense_budget
