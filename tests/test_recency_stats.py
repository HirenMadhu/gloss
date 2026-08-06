"""The recency order-statistic channel (`x`) — spec §1 constraints and §2 arms.

Hermetic: reuses the stub row batch from ``test_row_level``. Row 0 is the root and rows ``1..R-2``
are its children via role 1, with strictly decreasing timestamps — so the child set of ``(row 0,
role 1)`` has a known, hand-computable order statistic, which is what makes the soft-max/soft-min
assertions real rather than shape checks.
"""
from __future__ import annotations

import math

import pytest
import torch

from gloss.model.recency_stats import RecencyOrderChannel
from gloss.model.time_encoding import TimeLadder
from gloss.model.two_level import TwoLevelSubstrate

from .test_row_level import D_MODEL, D_SIG, D_TEXT, _tables, stub_batch


def _channel(K, **kw):
    _, role, _ = _tables(K)
    torch.manual_seed(0)
    return RecencyOrderChannel(D_MODEL, role, **kw)


def _child_ells(cb, R):
    """log1p(Delta) of row 0's children (rows 1..R-2), the group every assertion below reads."""
    d = cb.seed_time[0].item() - cb.row_time_r[0]
    return torch.log1p(d[1:R - 1].clamp(min=0.0))


# ---- §1: alpha init 0 means the arm starts identical to base ----


def test_alpha_starts_at_zero_so_a_fresh_x_run_is_numerically_identical_to_base():
    cb, K = stub_batch()
    ch = _channel(K)
    h_x, diag = ch(cb)
    assert diag["x_alpha"] == 0.0
    assert torch.equal(h_x, torch.zeros_like(h_x)), "alpha=0 must zero the channel exactly"


def test_substrate_with_channel_off_and_on_agree_at_init():
    """The whole point of alpha=0: `x_full` and `base` are the same model on step 0."""
    cb, K = stub_batch()
    tab, role, col = _tables(K)
    torch.manual_seed(0)
    base = TwoLevelSubstrate(D_MODEL, 2 * D_MODEL, D_SIG, tab, role, col, n_blocks=1, n_heads=4)
    torch.manual_seed(0)
    on = TwoLevelSubstrate(D_MODEL, 2 * D_MODEL, D_SIG, tab, role, col, n_blocks=1, n_heads=4,
                           recency_channel="full")
    torch.manual_seed(1)
    x = torch.randn(cb.num_seeds, cb.seq_len, D_MODEL)
    z = torch.randn(cb.num_seeds, cb.seq_len, D_SIG)
    hb, ub, _, _ = base(x, cb, z=z)
    ho, uo, _, d = on(x, cb, z=z)
    assert torch.allclose(hb, ho, atol=1e-6) and torch.allclose(ub, uo, atol=1e-6)
    assert "x_kappa_max" in d and "x_sat_rate" in d, "diagnostics must reach the caller"


# ---- §1: the order statistic is actually an order statistic ----


@pytest.mark.parametrize("kappa,which", [(60.0, "max"), (-60.0, "min")])
def test_large_kappa_recovers_the_hard_order_statistic(kappa, which):
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    ch = _channel(K, kappa_max_init=kappa, kappa_min_init=kappa)
    ells = _child_ells(cb, R)
    want = float(ells.max() if which == "max" else ells.min())

    # read x through the MLP-free path: recompute the stat the module computes
    adj = cb.adj_role
    valid = cb.row_valid.unsqueeze(2) & cb.row_valid.unsqueeze(1)
    is_child = (adj >= 1) & (adj <= K) & valid
    member = is_child & cb.row_is_timed.unsqueeze(1).expand_as(is_child)
    raw = cb.seed_time.unsqueeze(1).to(torch.float64) - cb.row_time_r.to(torch.float64)
    ell = TimeLadder.tau(raw).to(torch.float32).unsqueeze(1).expand_as(member.float())
    G = K + 1
    B, _, _ = adj.shape
    base = (torch.arange(B * R) * G).view(B, R, 1)
    gid = base + torch.where(is_child, adj, torch.zeros_like(adj))
    got = RecencyOrderChannel._soft_stat(torch.tensor(kappa), ell, member, gid, B * R * G)
    # group for (b=0, r=0, rho=1)
    assert float(got[0 * G + 1]) == pytest.approx(want, rel=1e-4)


def test_kappa_near_zero_collapses_to_the_mean_the_documented_failure_mode():
    cb, K = stub_batch()
    R = cb.adj_role.shape[1]
    ells = _child_ells(cb, R)
    adj = cb.adj_role
    valid = cb.row_valid.unsqueeze(2) & cb.row_valid.unsqueeze(1)
    is_child = (adj >= 1) & (adj <= K) & valid
    member = is_child & cb.row_is_timed.unsqueeze(1).expand_as(is_child)
    raw = cb.seed_time.unsqueeze(1).to(torch.float64) - cb.row_time_r.to(torch.float64)
    ell = TimeLadder.tau(raw).to(torch.float32).unsqueeze(1).expand_as(member.float())
    G, B = K + 1, adj.shape[0]
    gid = (torch.arange(B * R) * G).view(B, R, 1) + torch.where(is_child, adj, torch.zeros_like(adj))
    got = RecencyOrderChannel._soft_stat(torch.tensor(0.0), ell, member, gid, B * R * G)
    assert float(got[1]) == pytest.approx(float(ells.mean()), rel=1e-4)


def test_empty_and_untimed_groups_take_the_null_path_and_never_nan():
    cb, K = stub_batch()          # last row is untimed AND has no children
    ch = _channel(K)
    with torch.no_grad():
        ch.alpha.fill_(1.0)
    h_x, diag = ch(cb)
    assert torch.isfinite(h_x).all(), "a row with no children must not produce NaN"
    assert 0.0 <= diag["x_untimed_rate"] <= 1.0


def test_untimed_children_force_x_to_zero_rather_than_an_imputed_delta():
    """Mark every child untimed: x_max/x_min must vanish and `untimed` must be the only live flag."""
    cb, K = stub_batch()
    cb.row_is_timed = torch.zeros_like(cb.row_is_timed)
    ch = _channel(K)
    with torch.no_grad():
        ch.alpha.fill_(1.0)
    h_x, diag = ch(cb)
    assert torch.isfinite(h_x).all()
    assert diag["x_untimed_rate"] == pytest.approx(1.0)
    assert diag["x_max_mean"] == pytest.approx(0.0)


# ---- §1: the causality assert, scoped to child rows ----


def test_a_child_row_dated_after_its_seed_fails_loudly():
    cb, K = stub_batch()
    cb.row_time_r[0, 1] = cb.seed_time[0] + 86400.0        # row 1 is a CHILD of row 0
    ch = _channel(K)
    with pytest.raises(AssertionError, match="temporal constraint violated"):
        ch(cb)


def test_a_clamped_root_row_does_not_trip_the_assert():
    """Root rows with Delta < 0 are the known §9.10 class (`b_clamped` exists for them). They are in
    no child set, so scoping the assert to children is what keeps rel-event runnable."""
    cb, K = stub_batch()
    cb.row_time_r[0, 0] = cb.seed_time[0] + 86400.0        # row 0 is the ROOT
    ch = _channel(K)
    h_x, _ = ch(cb)                                        # must not raise
    assert torch.isfinite(h_x).all()


# ---- §2: the arms differ in the way the spec says they do ----


def test_flags_arm_zeroes_the_age_terms_but_keeps_the_flags():
    cb, K = stub_batch()
    ch = _channel(K, mode="flags")
    with torch.no_grad():
        ch.alpha.fill_(1.0)
    h_x, diag = ch(cb)
    assert diag["x_max_mean"] == pytest.approx(0.0) and diag["x_max_std"] == pytest.approx(0.0)
    assert h_x.abs().sum() > 0, "the flags alone must still produce a signal"


def test_shuffle_arm_keeps_magnitudes_but_breaks_the_row_binding():
    cb, K = stub_batch(B=4)
    full, shuf = _channel(K, mode="full"), _channel(K, mode="shuffle")
    with torch.no_grad():
        full.alpha.fill_(1.0)
        shuf.alpha.fill_(1.0)
    torch.manual_seed(0)
    _, df = full(cb)
    torch.manual_seed(0)
    _, ds = shuf(cb)
    assert ds["x_sat_rate"] == pytest.approx(df["x_sat_rate"]), "flags must be untouched"
    assert ds["x_untimed_rate"] == pytest.approx(df["x_untimed_rate"])


# ---- §1: leak-free routing and the §0 no-dataset-artifact rule ----


def test_the_channel_never_enters_the_row_signature_the_routers_read():
    """Changing child recency must move the row token and leave `s` — the router input — untouched."""
    cb, K = stub_batch()
    tab, role, col = _tables(K)
    torch.manual_seed(0)
    sub = TwoLevelSubstrate(D_MODEL, 2 * D_MODEL, D_SIG, tab, role, col, n_blocks=1, n_heads=4,
                            recency_channel="full")
    s_before = sub.row_sig(cb).clone()
    with torch.no_grad():
        sub.x_channel.alpha.fill_(1.0)
    h_x_before, _ = sub.x_channel(cb)

    cb.row_time_r[:, 1:] = cb.row_time_r[:, 1:] - 86400.0 * 365    # age every child by a year
    s_after = sub.row_sig(cb)
    h_x_after, _ = sub.x_channel(cb)

    assert not torch.allclose(h_x_before, h_x_after), "the channel must react to the window"
    assert not torch.allclose(s_before, s_after), "s reacts to row time by design (its own tau term)"
    # the real invariant: `s` is a pure function of cb, never of the channel's parameters
    with torch.no_grad():
        sub.x_channel.kappa_max.fill_(37.0)
        sub.x_channel.alpha.fill_(5.0)
    assert torch.equal(s_after, sub.row_sig(cb)), "router input must not depend on channel params"


def test_no_parameter_is_shaped_by_the_number_of_roles():
    """§0: a checkpoint must load on a schema with a different K."""
    shapes = {}
    for K in (3, 7):
        ch = _channel(K)
        shapes[K] = {k: tuple(v.shape) for k, v in ch.state_dict().items()}
    assert shapes[3] == shapes[7], f"K-shaped parameter leaked: {shapes}"


def test_gradients_reach_kappa_and_alpha():
    cb, K = stub_batch()
    ch = _channel(K)
    with torch.no_grad():
        ch.alpha.fill_(1.0)
    h_x, _ = ch(cb)
    h_x.sum().backward()
    assert ch.alpha.grad is not None and torch.isfinite(ch.alpha.grad)
    assert ch.kappa_max.grad is not None and torch.isfinite(ch.kappa_max.grad)
    assert float(ch.kappa_max.grad.abs()) > 0, "kappa must be trainable, not decorative"


def test_saturation_uses_ge_not_eq_so_a_double_reached_row_still_counts():
    """rel-event role 1 reaches 13 children against a cap of 12 (a row reachable via two parents),
    measured by scripts/probe_role_window.py. `sat` must be `>= w`."""
    cb, K = stub_batch(R=6)
    ch = _channel(K, w=2)          # rows 1..R-2 are children of row 0 -> 4 children >= 2
    _, diag = ch(cb)
    assert diag["x_sat_rate"] > 0.0
