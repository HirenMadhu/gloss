"""Row-level modules — changes.md §3.3–§3.7.

Hermetic: a hand-built stub batch, no relbench, no conftest fixtures, no GPU. The stub carries only
the fields the row modules read, so these tests exercise the operators rather than the collate path
(`tests/test_row_graph.py` owns that).

The load-bearing test here is :func:`test_no_parameter_shape_depends_on_role_count` — §0's
no-dataset-artifact rule at the row level. If `K` ever sizes a parameter, a checkpoint cannot load on
an unseen schema, which is the whole point of the two-level design.
"""
from __future__ import annotations

import types

import pytest
import torch

from gloss.model.row_level import (
    Broadcast,
    RowAttention,
    RowMoE,
    RowPool,
    RowSignature,
)
from gloss.model.time_encoding import TimeLadder

D_MODEL, D_SIG, D_TEXT, N_HEADS = 32, 16, 24, 4


def stub_batch(B=2, R=4, S=8, K=3, *, n_tables=3, empty_row=True, seed=0):
    """A minimal CellBatch-like namespace with the row fields the §3.3–3.7 modules read.

    Row `r` owns cells `[r*2, r*2+1]` for `r < R-1`; the last row deliberately owns NO cells and has
    NO neighbours (only its self-loop) when `empty_row=True`, which is the degenerate case that
    produces NaN if masking is done carelessly.
    """
    g = torch.Generator().manual_seed(seed)
    self_id = 2 * K + 1

    cell_row = torch.arange(S, device=None) // 2
    cell_row = cell_row.clamp_max(R - 1).unsqueeze(0).repeat(B, 1)
    is_padding = torch.zeros(B, S, dtype=torch.bool)
    if empty_row:
        # nothing maps to the last row, and the tail cells are padding
        cell_row[:, -2:] = 0
        is_padding[:, -2:] = True

    adj = torch.zeros(B, R, R, dtype=torch.long)
    for r in range(R):
        adj[:, r, r] = self_id
    # a small tree: row 0 is root, rows 1..R-2 are children of 0 via role 1
    for r in range(1, R - 1):
        adj[:, r, 0] = K + 1          # 0 is a PARENT of r
        adj[:, 0, r] = 1              # r is a CHILD of 0

    row_valid = torch.ones(B, R, dtype=torch.bool)
    row_is_root = torch.zeros(B, R, dtype=torch.bool)
    row_is_root[:, 0] = True
    row_is_timed = torch.ones(B, R, dtype=torch.bool)
    row_is_timed[:, -1] = False       # exercise the b_untimed path

    # Row timestamps must VARY across rows. With a constant timestamp every theta is equal, the
    # relative angle theta_i - theta_j is 0, and a uniform rotation of all q,k preserves inner
    # products — so RoPE is provably inert and any test of it passes vacuously. (That inertness is
    # itself the §6 relative-only property; it just makes a constant-time fixture useless here.)
    seed_t = 2.0e9
    row_t = seed_t - (torch.arange(R, dtype=torch.float64) + 1) * 86400.0 * 30
    row_time_r = row_t.unsqueeze(0).repeat(B, 1)

    # per-CELL fields: the cell level (two_level.CellAttention, build_relational_masks) reads these,
    # so the stub carries them even though the §3.3-3.7 row modules do not. A cell's time is its
    # ROW's time, matching the collate contract.
    max_fk = 2
    cell_time = torch.gather(row_time_r, 1, cell_row)
    cell_timed = ~is_padding
    f2p = torch.full((B, S, max_fk), -1, dtype=torch.long)
    f2p[:, 2:4, 0] = 0                     # rows 1..2's cells reference row 0 (the root)
    f2p[:, 4:6, 0] = 0

    return types.SimpleNamespace(
        num_seeds=B, seq_len=S, max_fk=max_fk,
        cell_row=cell_row, is_padding=is_padding,
        node_idxs=cell_row,                # row slot == node idx, matching the collate contract
        row_time=cell_time, is_timed=cell_timed,
        is_seed_cell=(cell_row == 0) & ~is_padding,
        f2p_nbr_idxs=f2p,
        col_idxs=torch.randint(0, 5, (B, S), generator=g),
        row_valid=row_valid,
        row_table=torch.randint(0, n_tables, (B, R), generator=g),
        row_in_role=torch.randint(0, K + 1, (B, R), generator=g),
        row_hop=torch.randint(0, 3, (B, R), generator=g),
        row_is_root=row_is_root,
        row_is_timed=row_is_timed,
        row_time_r=row_time_r,
        seed_time=torch.full((B,), seed_t, dtype=torch.float64),
        adj_role=adj,
    ), K


def _tables(K, n_tables=3):
    torch.manual_seed(0)
    return (torch.randn(n_tables, D_TEXT),          # table_name_emb
            torch.randn(K + 1, D_TEXT),             # role_name_emb, row 0 = FK_NONE
            torch.randn(8, D_TEXT))                 # col_name_emb


def _sig(K, ladder=None):
    tab, role, _ = _tables(K)
    return RowSignature(tab, role, ladder or TimeLadder(), d_sig=D_SIG)


# ---- shapes and the degenerate cases ----


def test_row_signature_shape_and_finiteness():
    cb, K = stub_batch()
    s = _sig(K)(cb)
    assert s.shape == (cb.num_seeds, cb.adj_role.shape[1], D_SIG)
    assert torch.isfinite(s).all()


def test_row_signature_is_value_free():
    """It reads table / role / hop / time only — no cell values enter. Changing `col_idxs` (a proxy
    for cell content reaching the row) must not move `s`."""
    cb, K = stub_batch()
    sig = _sig(K)
    a = sig(cb)
    cb.col_idxs = (cb.col_idxs + 1) % 5
    assert torch.equal(a, sig(cb))


@pytest.mark.parametrize("mode", ["mean", "signature", "hidden", "hybrid"])
def test_row_pool_all_query_arms_run(mode):
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    _, _, col = _tables(K)
    pool = RowPool(D_MODEL, D_SIG, col, slots=4, mode=mode)
    h = torch.randn(B, cb.seq_len, D_MODEL)
    u = torch.zeros(B, R, D_MODEL)
    s = torch.randn(B, R, D_SIG)
    out = pool(h, u, s, cb)
    assert out.shape == (B, R, D_MODEL)
    assert torch.isfinite(out).all(), f"{mode} produced non-finite output"


def test_row_pool_softmax_is_scoped_to_a_rows_own_cells():
    """Row `r` must receive nothing from cells of other rows, and nothing from padding.

    Verified behaviourally: perturbing ONLY the cells of row 1 must leave row 2's token unchanged.
    """
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    _, _, col = _tables(K)
    torch.manual_seed(0)
    pool = RowPool(D_MODEL, D_SIG, col, slots=2, mode="hybrid")
    h = torch.randn(B, cb.seq_len, D_MODEL)
    u = torch.randn(B, R, D_MODEL)
    s = torch.randn(B, R, D_SIG)

    base = pool(h, u, s, cb)
    h2 = h.clone()
    row1 = cb.cell_row[0] == 1
    h2[0, row1] += 5.0                                   # perturb only row 1's cells
    after = pool(h2, u, s, cb)

    moved = (after - base).abs().amax(dim=-1)            # [B,R]
    assert moved[0, 1] > 1e-6, "row 1 should react to its own cells"
    for r in range(R):
        if r != 1:
            assert moved[0, r] < 1e-6, f"row {r} leaked from row 1's cells"


def test_row_pool_ignores_padding_cells():
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    _, _, col = _tables(K)
    torch.manual_seed(0)
    pool = RowPool(D_MODEL, D_SIG, col, slots=2, mode="hybrid")
    h = torch.randn(B, cb.seq_len, D_MODEL)
    u, s = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)

    base = pool(h, u, s, cb)
    h2 = h.clone()
    h2[:, cb.is_padding[0]] += 1e3                       # garbage in the padding slots
    assert torch.allclose(base, pool(h2, u, s, cb), atol=1e-6)


@pytest.mark.parametrize("time_bias", ["rope", "none", "fixed_basis"])
@pytest.mark.parametrize("role_bias", ["name_derived", "none"])
def test_row_attention_all_arms_run_and_stay_finite(time_bias, role_bias):
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    _, role, _ = _tables(K)
    att = RowAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS,
                       role_bias=role_bias, time_bias=time_bias)
    u, s = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)
    out, diag = att(u, s, cb)
    assert out.shape == (B, R, D_MODEL)
    assert torch.isfinite(out).all(), "non-finite — a fully-masked row is the usual cause"
    assert torch.isfinite(diag["row_attn_entropy"])


def test_self_loop_means_no_row_is_ever_fully_masked():
    """A row with no neighbours still has its self-loop, so softmax never sees an all -inf row."""
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    # strip every edge except the self-loops
    self_id = 2 * K + 1
    cb.adj_role = torch.zeros_like(cb.adj_role)
    for r in range(R):
        cb.adj_role[:, r, r] = self_id
    _, role, _ = _tables(K)
    att = RowAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS)
    out, _ = att(torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG), cb)
    assert torch.isfinite(out).all()


def test_masked_pairs_get_no_attention():
    """`adj_role == 0` must be inadmissible: perturbing a non-neighbour cannot move the output."""
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    _, role, _ = _tables(K)
    torch.manual_seed(0)
    att = RowAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS, time_bias="none")
    u, s = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)

    # rows 1 and 2 are siblings: both children of 0, no edge between them
    assert int(cb.adj_role[0, 1, 2]) == 0
    base, _ = att(u, s, cb)
    u2 = u.clone()
    u2[0, 2] += 10.0
    after, _ = att(u2, s, cb)
    assert (after[0, 1] - base[0, 1]).abs().max() < 1e-6, "row 1 attended to a masked sibling"


# ---- §0 / §6: the artifact guard at the row level ----


def test_no_parameter_shape_depends_on_role_count():
    """Two schemas with DIFFERENT K must give byte-identical `state_dict` shapes.

    This is the §0 rule made testable. `γ` is a bilinear form on the frozen role-name embedding plus
    three learned directions, so `K` indexes DATA only. The earlier `γ ∈ R^{2K+2}` design would fail
    this, and would be undefined on an unseen database rather than merely miscalibrated.
    """
    shapes = []
    for K in (3, 11):
        _, role, _ = _tables(K)
        att = RowAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS)
        shapes.append({k: tuple(v.shape) for k, v in att.state_dict().items()})
    assert shapes[0] == shapes[1], f"K leaked into a weight shape: {shapes}"

    sig_shapes = []
    for K in (3, 11):
        sig_shapes.append({k: tuple(v.shape) for k, v in _sig(K).state_dict().items()})
    assert sig_shapes[0] == sig_shapes[1], f"K leaked into RowSignature: {sig_shapes}"


def test_frozen_name_tables_are_not_in_state_dict():
    """Non-persistent buffers: a K-shaped tensor in `state_dict` would break cross-schema loading."""
    _, role, _ = _tables(5)
    att = RowAttention(D_MODEL, D_SIG, role, TimeLadder(), n_heads=N_HEADS)
    assert not any("role_name_emb" in k for k in att.state_dict())
    sig = _sig(5)
    assert not any(n in k for k in sig.state_dict() for n in ("table_name_emb", "role_name_emb"))


def test_checkpoint_from_one_schema_loads_on_another():
    """The cheap proxy for the deferred LODO run (§6)."""
    _, role_a, _ = _tables(3)
    _, role_b, _ = _tables(11)
    a = RowAttention(D_MODEL, D_SIG, role_a, TimeLadder(), n_heads=N_HEADS)
    b = RowAttention(D_MODEL, D_SIG, role_b, TimeLadder(), n_heads=N_HEADS)
    b.load_state_dict(a.state_dict(), strict=True)       # must not raise


# ---- row MoE ----


def test_row_moe_gates_sum_to_one_over_topk():
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_experts=4, k=2)
    g = moe.gates(torch.randn(B, R, D_SIG))
    assert torch.allclose(g.sum(-1), torch.ones(B, R), atol=1e-5)
    assert int((g > 0).sum(-1).max()) == 2, "top-k support must be exactly k"


@pytest.mark.parametrize("use_shared", [False, True])
def test_row_moe_dense_combine_equals_weighted_expert_sum(use_shared):
    """Dense combine == gate-weighted routed sum, PLUS the ungated shared expert when enabled."""
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    torch.manual_seed(0)
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_experts=4, k=2, use_shared=use_shared)
    u, z = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)
    out, _, _ = moe(u, z, cb)

    g = moe.gates(z)
    x = moe.norm(u)
    ref = sum(g[..., e:e + 1] * moe.experts[e](x) for e in range(4))
    if use_shared:
        ref = ref + moe.shared(x)          # always-on, NOT multiplied by any gate
    assert torch.allclose(out - u, ref, atol=1e-5)


def test_row_moe_ortho_loss_finite_and_positive():
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_experts=4, k=2)
    v = moe.ortho_loss()
    assert torch.isfinite(v) and float(v) > 0


def test_row_moe_balance_loss_is_minimal_for_uniform_usage():
    """`M·Σ f_e p_e` = 1 at perfectly uniform usage and larger when collapsed."""
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_experts=4, k=4)
    valid = torch.ones(1, 4, dtype=torch.bool)

    uniform = torch.full((1, 4, 4), 0.25)
    collapsed = torch.zeros(1, 4, 4)
    collapsed[..., 0] = 1.0

    assert float(moe.balance_loss(uniform, valid)) == pytest.approx(1.0, abs=1e-5)
    assert float(moe.balance_loss(collapsed, valid)) > float(moe.balance_loss(uniform, valid))


def test_row_moe_padding_rows_do_not_vote_in_balance():
    cb, K = stub_batch()
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_experts=4, k=2)
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    g = moe.gates(torch.randn(B, R, D_SIG))

    all_valid = torch.ones(B, R, dtype=torch.bool)
    some_valid = all_valid.clone()
    some_valid[:, -1] = False
    # the two must differ, i.e. the mask is actually applied rather than ignored
    assert float(moe.balance_loss(g, all_valid)) != float(moe.balance_loss(g, some_valid))


# ---- broadcast + grad flow ----


@pytest.mark.parametrize("mode", ["additive", "film", "none"])
def test_broadcast_shapes(mode):
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    bc = Broadcast(D_MODEL, mode=mode)
    h = torch.randn(B, cb.seq_len, D_MODEL)
    out = bc(h, torch.randn(B, R, D_MODEL), cb)
    assert out.shape == h.shape and torch.isfinite(out).all()


def test_grad_reaches_router_signature_wtau_and_b_untimed():
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    ladder = TimeLadder()
    sig = _sig(K, ladder)
    _, role, col = _tables(K)
    pool = RowPool(D_MODEL, D_SIG, col, slots=2, mode="hybrid")
    att = RowAttention(D_MODEL, D_SIG, role, ladder, n_heads=N_HEADS)
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG)

    h = torch.randn(B, cb.seq_len, D_MODEL)
    s = sig(cb)
    u = pool(h, torch.zeros(B, R, D_MODEL), s, cb)
    u, _ = att(u, s, cb)
    u, aux, _ = moe(u, s, cb)
    (u.sum() + aux).backward()

    for name, p in (("router w_g", moe.w_g), ("log_T", moe.log_T),
                    ("sig w_tau", sig.w_tau.weight), ("sig w_tab", sig.w_tab.weight),
                    ("gamma v_head", att.v_head), ("b_untimed", ladder.b_untimed)):
        assert p.grad is not None, f"no grad reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad at {name}"


# ---- shared + routed row experts ----


def test_row_moe_is_shared_plus_routed_by_default():
    moe = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG)
    assert moe.shared is not None, "row experts should be shared+routed by default"


def test_shared_expert_is_always_on_regardless_of_gates():
    """The shared expert must contribute even for a row whose gates concentrate elsewhere."""
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    torch.manual_seed(0)
    shared = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=True)
    u, z = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)

    out, _, _ = shared(u, z, cb)
    g = shared.gates(z)
    x = shared.norm(u)
    routed = sum(g[..., e:e + 1] * shared.experts[e](x) for e in range(shared.num_experts))
    # output = u + routed + shared(x); the shared term is NOT gated
    assert torch.allclose(out - u - routed, shared.shared(x), atol=1e-5)


def test_routed_only_still_available_for_the_ablation():
    cb, K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    routed_only = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=False)
    assert routed_only.shared is None
    out, aux, _ = routed_only(torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG), cb)
    assert torch.isfinite(out).all() and torch.isfinite(aux)


def test_shared_expert_adds_parameters_but_no_dataset_shape():
    a = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=False)
    b = RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=True)
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    assert nb > na, "shared expert should add parameters"
    # one extra SwiGLU, nothing sized by a dataset id
    assert all("shared" not in k or v.shape[0] in (D_MODEL, 4 * D_MODEL, 2 * 4 * D_MODEL)
               for k, v in b.state_dict().items())
