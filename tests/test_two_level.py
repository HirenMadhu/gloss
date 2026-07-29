"""The two-level substrate — changes.md §3.2, §3.9, and the §3.8 row-token head.

Hermetic: reuses the stub batch from ``test_row_level`` so nothing here depends on relbench or on
``conftest.py``. These tests are about COMPOSITION — that the six sublayers wire together, that every
phase's config is reachable, and that Phase 0a can be made to behave like RT at the cell level.
"""
from __future__ import annotations

import pytest
import torch

from gloss.model.heads import EntityHead, RowTokenHead
from gloss.model.two_level import CellAttention, TwoLevelBlock, TwoLevelSubstrate
from gloss.model.time_encoding import TimeLadder

from .test_row_level import D_MODEL, D_SIG, D_TEXT, _tables, stub_batch

D_FF, N_HEADS, N_BLOCKS = 2 * D_MODEL, 4, 2


def _substrate(K, **kw):
    tab, role, col = _tables(K)
    torch.manual_seed(0)
    return TwoLevelSubstrate(D_MODEL, D_FF, D_SIG, tab, role, col,
                             n_blocks=N_BLOCKS, n_heads=N_HEADS, **kw)


def _inputs(cb):
    B, S = cb.num_seeds, cb.seq_len
    torch.manual_seed(1)
    return torch.randn(B, S, D_MODEL), torch.randn(B, S, D_SIG)


# ---- composition ----


def test_substrate_forward_shapes_and_finiteness():
    cb, K = stub_batch()
    sub = _substrate(K)
    x, z = _inputs(cb)
    h, u, aux, diag = sub(x, cb, z=z)

    assert h.shape == x.shape
    assert u.shape == (cb.num_seeds, cb.adj_role.shape[1], D_MODEL)
    assert torch.isfinite(h).all() and torch.isfinite(u).all()
    assert torch.isfinite(aux) and aux.ndim == 0


def test_aux_is_split_by_level():
    """A collapse at one level must not be maskable by the other — §3.7's recorded consequence."""
    cb, K = stub_batch()
    sub = _substrate(K, cell_ffn="moe", row_ffn="moe")
    x, z = _inputs(cb)
    _, _, aux, diag = sub(x, cb, z=z)

    assert "aux_cell" in diag and "aux_row" in diag
    assert float(diag["aux_cell"]) > 0, "cell MoE should contribute ortho loss"
    assert float(diag["aux_row"]) > 0, "row MoE should contribute ortho + balance"
    assert float(aux) == pytest.approx(float(diag["aux_cell"]) + float(diag["aux_row"]), rel=1e-5)


def test_row_signature_is_computed_once_and_shared():
    """§3.3 says once per forward. Blocks must not each rebuild it."""
    cb, K = stub_batch()
    sub = _substrate(K)
    calls = []
    orig = sub.row_sig.forward
    sub.row_sig.forward = lambda b: (calls.append(1), orig(b))[1]
    x, z = _inputs(cb)
    sub(x, cb, z=z)
    assert len(calls) == 1, f"row signature rebuilt {len(calls)}x for {N_BLOCKS} blocks"


def test_diagnostics_are_returned_per_block():
    cb, K = stub_batch()
    sub = _substrate(K, row_ffn="moe")
    x, z = _inputs(cb)
    _, _, _, diag = sub(x, cb, z=z)
    assert len(diag["blocks"]) == N_BLOCKS
    b0 = diag["blocks"][0]
    for key in ("row_expert_usage", "row_router_norms", "row_ortho", "row_balance", "row_T",
                "gamma_abs_mean"):
        assert key in b0, f"missing §7 instrumentation key {key!r}"


# ---- every phase's config must be reachable ----


@pytest.mark.parametrize("cell_attention", ["full", "four_mask"])
@pytest.mark.parametrize("cell_rope_time", [True, False])
def test_phase0_cell_level_arms(cell_attention, cell_rope_time):
    """Phase 0a needs `four_mask` + no RoPE (RT at the cell level); Phase 0b flips to `full`."""
    cb, K = stub_batch()
    sub = _substrate(K, cell_attention=cell_attention, cell_rope_time=cell_rope_time)
    x, z = _inputs(cb)
    h, u, _, _ = sub(x, cb, z=z)
    assert torch.isfinite(h).all() and torch.isfinite(u).all()


@pytest.mark.parametrize("row_ffn,cell_ffn", [("moe", "moe"), ("dense", "moe"), ("moe", "dense")])
def test_phase4_moe_arms(row_ffn, cell_ffn):
    """Phase 4: r1 = both levels (the design), r0 = cell only, r2 = row only."""
    cb, K = stub_batch()
    sub = _substrate(K, row_ffn=row_ffn, cell_ffn=cell_ffn)
    x, z = _inputs(cb)
    h, u, aux, _ = sub(x, cb, z=z)
    assert torch.isfinite(h).all() and torch.isfinite(aux)


@pytest.mark.parametrize("pool_query", ["mean", "signature", "hidden", "hybrid"])
def test_phase3_pool_arms(pool_query):
    cb, K = stub_batch()
    sub = _substrate(K, pool_query=pool_query)
    x, z = _inputs(cb)
    h, _, _, _ = sub(x, cb, z=z)
    assert torch.isfinite(h).all()


@pytest.mark.parametrize("time_bias", ["rope", "none", "fixed_basis"])
@pytest.mark.parametrize("role_bias", ["name_derived", "none"])
def test_phase1_and_2_row_bias_arms(time_bias, role_bias):
    cb, K = stub_batch()
    sub = _substrate(K, time_bias=time_bias, role_bias=role_bias)
    x, z = _inputs(cb)
    h, _, _, _ = sub(x, cb, z=z)
    assert torch.isfinite(h).all()


@pytest.mark.parametrize("broadcast", ["additive", "film", "none"])
def test_broadcast_arms(broadcast):
    cb, K = stub_batch()
    sub = _substrate(K, broadcast=broadcast)
    x, z = _inputs(cb)
    h, _, _, _ = sub(x, cb, z=z)
    assert torch.isfinite(h).all()


# ---- cell attention specifics ----


def test_cell_attention_ignores_padding():
    cb, K = stub_batch()
    B, S = cb.num_seeds, cb.seq_len
    torch.manual_seed(0)
    att = CellAttention(D_MODEL, N_HEADS, TimeLadder(), rope_time=True)
    x = torch.randn(B, S, D_MODEL)

    base = att(x, cb)
    x2 = x.clone()
    x2[:, cb.is_padding[0]] += 1e3
    after = att(x2, cb)
    real = ~cb.is_padding[0]
    assert torch.allclose(base[:, real], after[:, real], atol=1e-4), "padding leaked into real cells"


def test_cell_attention_rope_actually_changes_scores():
    """Guard against RoPE being silently inert — a wiring bug that no shape test would catch."""
    cb, K = stub_batch()
    B, S = cb.num_seeds, cb.seq_len
    torch.manual_seed(0)
    x = torch.randn(B, S, D_MODEL)

    torch.manual_seed(0)
    on = CellAttention(D_MODEL, N_HEADS, TimeLadder(), rope_time=True)
    torch.manual_seed(0)
    off = CellAttention(D_MODEL, N_HEADS, TimeLadder(), rope_time=False)
    assert not torch.allclose(on(x, cb), off(x, cb), atol=1e-5)


def test_all_padding_row_does_not_produce_nan():
    cb, K = stub_batch()
    B, S = cb.num_seeds, cb.seq_len
    cb.is_padding = torch.ones(B, S, dtype=torch.bool)
    att = CellAttention(D_MODEL, N_HEADS, TimeLadder())
    out = att(torch.randn(B, S, D_MODEL), cb)
    assert torch.isfinite(out).all()


# ---- §3.8 head ----


def test_row_token_head_reads_the_root_row():
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    torch.manual_seed(0)
    head = RowTokenHead(D_MODEL, out_dim=1)
    u = torch.randn(B, R, D_MODEL)

    base = head(u, cb)
    u2 = u.clone()
    u2[:, 1] += 10.0                                  # a NON-root row
    assert torch.allclose(base, head(u2, cb), atol=1e-6), "head read a non-root row"

    # NOTE perturb NON-uniformly. The head starts with LayerNorm, so adding a constant to every dim
    # of the pooled vector is removed by the mean subtraction — a `+= 10.0` here passes trivially and
    # tests nothing. Scaling changes the direction, which LayerNorm does not undo.
    u3 = u.clone()
    u3[:, 0] = u3[:, 0] * 3.0 + torch.randn(B, D_MODEL)
    assert not torch.allclose(base, head(u3, cb), atol=1e-4)


def test_both_head_modes_produce_the_same_shape():
    """`head.mode: seed_cells` must stay reachable for the Phase 0 parity check."""
    cb, K = stub_batch()
    B, R, S = cb.num_seeds, cb.adj_role.shape[1], cb.seq_len
    cb.is_seed_cell = cb.cell_row == 0
    row_head, cell_head = RowTokenHead(D_MODEL), EntityHead(D_MODEL)
    assert row_head(torch.randn(B, R, D_MODEL), cb).shape == \
        cell_head(torch.randn(B, S, D_MODEL), cb).shape == (B, 1)


# ---- §0 artifact guard at substrate level ----


def test_substrate_checkpoint_loads_across_schemas():
    """Different table/role counts, identical state_dict — the LODO proxy, end to end."""
    a, b = _substrate(3), _substrate(11)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sa == sb, "a training-set id leaked into a weight shape"
    b.load_state_dict(a.state_dict(), strict=True)


def test_grad_flows_through_both_levels():
    cb, K = stub_batch()
    sub = _substrate(K)
    x, z = _inputs(cb)
    x.requires_grad_(True)
    h, u, aux, _ = sub(x, cb, z=z)
    (h.sum() + u.sum() + aux).backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    named = dict(sub.named_parameters())
    for key in ("w_u.weight", "row_sig.w_tau.weight", "blocks.0.row_attn.v_head",
                "blocks.0.row_ffn.w_g"):
        p = named[key]
        assert p.grad is not None, f"no grad reached {key}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad at {key}"
