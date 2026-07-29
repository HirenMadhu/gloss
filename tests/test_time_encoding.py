"""The fixed time ladder (changes.md §3.1) and its §6 test list.

Hermetic: pure synthetic tensors — no relbench, no dataset, no cache, no network. The module has no data
dependency at all, by design (that *is* the no-dataset-artifact rule of §0), so neither does its test.

Everything here is float64. That is not fussiness: UNIX timestamps are ~1.7e9 s and, after the ×86400
rescale of the time-unit test, ~1.5e14 — float32 cannot even represent the differences. The invariance
under test is a property of the *encoding*, so the arithmetic around it must not be the thing that fails.
"""
from __future__ import annotations

import math

import torch

from gloss.model.time_encoding import TAU_MAX_PLAUSIBLE, TimeLadder

DAY = 86_400.0
YEAR = 365.25 * DAY
CENTURY = 100 * YEAR


def _qk(n: int = 12, d: int = 32, *, unit: bool = False, seed: int = 0):
    """Fixed synthetic q/k. ``unit=True`` row-normalises them, which bounds |logit| by 1."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, d, generator=g, dtype=torch.float64)
    k = torch.randn(n, d, generator=g, dtype=torch.float64)
    if unit:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
    return q, k


def _logits(ladder: TimeLadder, q, k, tau, *, scale: bool = True):
    """RoPE attention logits for fixed hidden states: ``R(θ)q · R(θ)k / sqrt(d)``."""
    th = ladder.theta(tau)
    qt, kt = ladder.rotate(q, th), ladder.rotate(k, th)
    out = qt @ kt.transpose(-1, -2)
    return out / math.sqrt(q.shape[-1]) if scale else out


# --------------------------------------------------------------------------- relative-only ---

def test_relative_only_shift_in_tau_leaves_logits_unchanged():
    """§6: the logit depends on τ_i, τ_j only through (τ_i − τ_j) — so the deleted μ_τ was redundant."""
    ladder = TimeLadder()
    q, k = _qk()
    tau = ladder.tau(torch.tensor([0.0, 1.0, 60.0, 3600.0, DAY, 30 * DAY, YEAR, 10 * YEAR,
                                   50 * YEAR, CENTURY, 1e3, 1e7]))
    base = _logits(ladder, q, k, tau)
    for shift in (0.5, 3.0, -0.25, 17.0):
        shifted = _logits(ladder, q, k, tau + shift)
        assert torch.allclose(shifted, base, atol=1e-5), f"shift {shift} moved the logits"


def test_rotation_score_is_a_function_of_the_angle_difference():
    """The algebraic reason for the above: R(θ_i)q · R(θ_j)k == R(θ_i − θ_j)q · k."""
    ladder = TimeLadder()
    q, k = _qk(n=1)
    ti = ladder.theta(torch.tensor([3.0]))
    tj = ladder.theta(torch.tensor([1.25]))
    both = (ladder.rotate(q, ti) * ladder.rotate(k, tj)).sum()
    diff = (ladder.rotate(q, ti - tj) * k).sum()
    assert torch.allclose(both, diff, atol=1e-10)


# ------------------------------------------------------------------- time-unit invariance ---

def test_time_unit_invariance_exact_regime_delta_ge_one_day():
    """§6 foundation-model test, stated in its regime.

    Multiply every timestamp by ``c ∈ {60, 86400}`` (seconds→minutes→days-worth of rescaling) and the
    RoPE logits are unchanged to 1e-5 **for Δ ≥ 1 day**, because
    ``τ_i − τ_j = log((1+Δ_i)/(1+Δ_j)) → log Δ_i − log Δ_j`` and the ``log c`` cancels.

    Exact for ``Δ ≫ 1s``; approximate near the ``+1`` floor (that regime is
    :func:`test_time_unit_invariance_degrades_near_the_plus_one_floor`, and it is correct behaviour, not
    a defect). ``q``/``k`` are row-normalised so ``|logit| ≤ 1`` and the absolute 1e-5 is a statement
    about a bounded quantity — the residual floor error scales with ‖q‖‖k‖, which is a property of the
    hidden states, not of the encoding.
    """
    ladder = TimeLadder()
    q, k = _qk(unit=True)
    delta = torch.tensor([DAY * (1.7 ** i) for i in range(12)], dtype=torch.float64)
    assert float(delta.min()) >= DAY
    base = _logits(ladder, q, k, ladder.tau(delta))
    for c in (60.0, 86_400.0):
        rescaled = _logits(ladder, q, k, ladder.tau(delta * c))
        assert torch.allclose(rescaled, base, atol=1e-5), f"unit rescale by {c} moved the logits"


def test_time_unit_invariance_residual_matches_the_plus_one_floor_bound():
    """The residual at Δ ≥ 1 day is exactly the ``+1`` floor, and it shrinks as Δ_min grows."""
    ladder = TimeLadder()
    omega_max = float(ladder.omega.max())

    def max_angle_error(delta_min_days: float, c: float) -> float:
        delta = torch.tensor([delta_min_days * DAY * (1.7 ** i) for i in range(8)],
                             dtype=torch.float64)
        a = ladder.theta(ladder.tau(delta))
        b = ladder.theta(ladder.tau(delta * c))
        rel_a = a.unsqueeze(0) - a.unsqueeze(1)          # θ_i − θ_j, the only thing RoPE sees
        rel_b = b.unsqueeze(0) - b.unsqueeze(1)
        return float((rel_a - rel_b).abs().max())

    for c in (60.0, 86_400.0):
        e_day = max_angle_error(1.0, c)
        assert e_day <= omega_max / DAY * 1.05, "residual exceeds the analytic 1/(1+Δ_min) bound"
        assert max_angle_error(30.0, c) < e_day / 10, "residual should shrink ~1/Δ_min"


def test_time_unit_invariance_degrades_near_the_plus_one_floor():
    """Documented non-invariance at Δ → 0 (asserted as *present*, never as absent — §6)."""
    ladder = TimeLadder()
    q, k = _qk(n=6, unit=True)
    delta = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    base = _logits(ladder, q, k, ladder.tau(delta))
    rescaled = _logits(ladder, q, k, ladder.tau(delta * 86_400.0))
    assert (rescaled - base).abs().max() > 1e-3


# ------------------------------------------------------------------------------- untimed ---

def test_untimed_theta_and_feats_are_zero():
    """§3.1: untimed rows get θ = 0 on every channel; the feature readout is all-zero too."""
    ladder = TimeLadder()
    tau = ladder.tau(torch.tensor([0.0, DAY, YEAR, 3.0 * YEAR]))
    is_timed = torch.tensor([True, False, False, True])
    th = ladder.theta(tau, is_timed)
    assert torch.count_nonzero(th[~is_timed]) == 0
    assert (th[is_timed][1] != 0).any()
    f = ladder.feats(tau, is_timed)
    assert f.shape == (4, ladder.feat_dim)          # 2*n_freq + the §9.10 clamped indicator
    assert torch.count_nonzero(f[~is_timed]) == 0  # untimed zeroes the sinusoids AND the flag


def test_untimed_is_not_equivalent_to_delta_zero():
    """θ = 0 alone would collapse untimed onto "Δ = 0, maximally recent"; the b_untimed flag is what
    keeps them apart. Both readouts must separate them."""
    ladder = TimeLadder()
    ladder.b_untimed.data.fill_(0.7)                 # zero-init is a no-prior default, not a no-op path
    q, k = _qk(n=4, unit=True)

    tau = torch.zeros(4, dtype=torch.float64)        # θ = 0 either way
    all_timed = torch.ones(4, dtype=torch.bool)
    some_untimed = torch.tensor([True, False, True, False])

    rope = _logits(ladder, q, k, tau)
    lg_timed = rope + ladder.untimed_bias(all_timed, all_timed)
    lg_untimed = rope + ladder.untimed_bias(some_untimed, some_untimed)
    assert not torch.allclose(lg_timed, lg_untimed), "untimed must not equal Δ = 0"

    # exactly the pairs touching an untimed endpoint moved, by exactly b_untimed
    moved = ~torch.isclose(lg_timed, lg_untimed)
    expected = ~(some_untimed.unsqueeze(-1) & some_untimed.unsqueeze(-2))
    assert torch.equal(moved, expected)
    assert torch.allclose((lg_untimed - lg_timed)[expected], torch.full((int(expected.sum()),), 0.7,
                                                                       dtype=torch.float64))

    # the feature readout separates them as well (zeros vs [0 ; 1])
    f_untimed = ladder.feats(tau, some_untimed)[1]
    f_delta0 = ladder.feats(tau, all_timed)[1]
    assert not torch.allclose(f_untimed, f_delta0)


def test_b_untimed_is_a_single_learned_scalar_with_a_live_gradient():
    ladder = TimeLadder()
    assert isinstance(ladder.b_untimed, torch.nn.Parameter)
    assert ladder.b_untimed.numel() == 1                 # one universal parameter, no dataset dependence
    is_timed = torch.tensor([True, False, True])
    ladder.untimed_bias(is_timed, is_timed).sum().backward()
    assert ladder.b_untimed.grad is not None and float(ladder.b_untimed.grad) != 0.0


# ------------------------------------------------------------------------ tau / ladder range ---

def test_tau_is_finite_and_in_the_universal_band():
    """§6: τ finite at Δ = 0 and at any large Δ; τ ∈ [0, 22] for any physically plausible Δ."""
    ladder = TimeLadder()
    assert float(ladder.tau(torch.tensor([0.0]))) == 0.0
    assert abs(float(ladder.tau(torch.tensor([1.0]))) - 0.6931) < 1e-3        # 1 second
    assert abs(float(ladder.tau(torch.tensor([CENTURY]))) - 21.9) < 0.1       # 1 century
    plausible = torch.tensor([0.0, 1.0, 60.0, DAY, YEAR, CENTURY], dtype=torch.float64)
    tau = ladder.tau(plausible)
    assert torch.isfinite(tau).all()
    assert float(tau.min()) >= 0.0 and float(tau.max()) <= TAU_MAX_PLAUSIBLE
    # negative Δ (clock skew / future rows) is clamped, never NaN
    assert float(ladder.tau(torch.tensor([-5.0]))) == 0.0
    # far outside anything physical: still finite, ladder included
    huge = ladder.tau(torch.tensor([1e18, torch.finfo(torch.float64).max], dtype=torch.float64))
    assert torch.isfinite(huge).all() and torch.isfinite(ladder.theta(huge)).all()
    assert torch.isfinite(ladder.feats(huge)).all()


def test_delta_seconds_clamps_and_keeps_float64_resolution():
    ladder = TimeLadder()
    seed = torch.tensor([1.7e9, 1.7e9], dtype=torch.float64)
    row = torch.tensor([1.7e9 - 1.0, 1.7e9 + 500.0], dtype=torch.float64)
    d = ladder.delta_seconds(seed, row)
    assert d.dtype == torch.float64
    assert float(d[0]) == 1.0 and float(d[1]) == 0.0     # a 1 s gap survives; the future row clamps
    assert torch.allclose(ladder.tau_from_times(seed, row), ladder.tau(d))


def test_ladder_is_log_spaced_over_the_configured_band():
    ladder = TimeLadder(n_freq=8, omega=(0.05, 5.0))
    w = ladder.omega
    assert w.shape == (8,)
    assert math.isclose(float(w[0]), 0.05, rel_tol=1e-6)
    assert math.isclose(float(w[-1]), 5.0, rel_tol=1e-6)
    ratios = (w[1:] / w[:-1]).tolist()
    assert max(ratios) - min(ratios) < 1e-9              # log-spaced
    # band design (§3.1): lowest freq monotonic across the whole τ ∈ [0, 22] band, no wraparound
    assert 2 * math.pi / float(w[0]) > TAU_MAX_PLAUSIBLE
    theta = ladder.theta(torch.linspace(0, TAU_MAX_PLAUSIBLE, 200, dtype=torch.float64))
    assert (theta[1:, 0] > theta[:-1, 0]).all()


# ------------------------------------------------------------------------------- rotation ---

def test_rotate_preserves_norm_and_leaves_the_tail_untouched():
    ladder = TimeLadder()
    x = torch.randn(3, 5, 32, dtype=torch.float64)
    th = ladder.theta(ladder.tau(torch.rand(3, 5, dtype=torch.float64) * 1e6))
    y = ladder.rotate(x, th, n_rot_dims=16)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y[..., 16:], x[..., 16:])                      # tail passes through
    assert torch.allclose(y[..., :16].norm(dim=-1), x[..., :16].norm(dim=-1))
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1))


def test_rotate_by_zero_angle_is_the_identity():
    ladder = TimeLadder()
    x = torch.randn(2, 4, 16, dtype=torch.float64)
    zero = ladder.theta(torch.zeros(2, 4, dtype=torch.float64))
    assert torch.allclose(ladder.rotate(x, zero), x)


def test_rotate_broadcasts_over_heads_and_rejects_bad_dims():
    ladder = TimeLadder()
    x = torch.randn(2, 3, 7, 32, dtype=torch.float64)            # [B, H, S, d_h]
    th = ladder.theta(ladder.tau(torch.rand(2, 7, dtype=torch.float64) * 1e6))
    y = ladder.rotate(x, th.unsqueeze(1), n_rot_dims=16)         # [B, 1, S, n_freq]
    assert y.shape == x.shape
    for bad in (15, 64, 2 * ladder.n_freq + 2):
        try:
            ladder.rotate(x, th.unsqueeze(1), n_rot_dims=bad)
        except ValueError:
            continue
        raise AssertionError(f"n_rot_dims={bad} should have raised")


def test_rotate_matches_a_naive_pairwise_reference():
    ladder = TimeLadder(n_freq=4)
    x = torch.randn(6, 8, dtype=torch.float64)
    th = ladder.theta(ladder.tau(torch.rand(6, dtype=torch.float64) * 1e6))
    y = ladder.rotate(x, th)
    ref = x.clone()
    for t in range(6):
        for p in range(4):
            a, b = x[t, 2 * p], x[t, 2 * p + 1]
            c, s = math.cos(float(th[t, p])), math.sin(float(th[t, p]))
            ref[t, 2 * p], ref[t, 2 * p + 1] = a * c - b * s, a * s + b * c
    assert torch.allclose(y, ref, atol=1e-12)


def test_feats_are_sin_then_cos_of_the_same_ladder():
    ladder = TimeLadder()
    tau = ladder.tau(torch.tensor([0.0, DAY, YEAR], dtype=torch.float64))
    th = ladder.theta(tau)
    f = ladder.feats(tau)
    n = ladder.n_freq
    assert torch.allclose(f[..., :n], torch.sin(th))
    assert torch.allclose(f[..., n:2 * n], torch.cos(th))
    # trailing channel is the §9.10 clamped indicator, 0 when no clamp info was supplied
    assert f.shape[-1] == ladder.feat_dim
    assert torch.count_nonzero(f[..., -1]) == 0


def test_feats_behind_a_learned_map_have_a_gradient_but_omega_does_not():
    """§3.1: the linear map is learned, the frequencies are not."""
    ladder = TimeLadder()
    w = torch.nn.Linear(ladder.feat_dim, 5, dtype=torch.float64)
    tau = ladder.tau(torch.tensor([DAY, YEAR], dtype=torch.float64))
    w(ladder.feats(tau, torch.tensor([True, True]))).sum().backward()
    assert w.weight.grad is not None and torch.isfinite(w.weight.grad).all()
    assert not ladder.omega.requires_grad


# ----------------------------------------------------- no dataset-specific artifact (§6 guard) ---

def test_omega_is_a_non_persistent_buffer_not_a_parameter():
    ladder = TimeLadder()
    sd = ladder.state_dict()
    assert "omega" not in sd, "omega must be persistent=False — a checkpointed ladder is an artifact"
    # Two learned scalars, both UNIVERSAL (no dataset shape): b_untimed (§3.1) and b_clamped
    # (§9.10, rows dated after their own seed). Anything else appearing here is an artifact.
    assert sorted(sd) == ["b_clamped", "b_untimed"], f"unexpected state_dict entries: {list(sd)}"
    names = {n for n, _ in ladder.named_parameters()}
    assert names == {"b_untimed", "b_clamped"}
    assert all(v.ndim == 0 for v in sd.values()), "both flags must be scalars, not per-dataset rows"
    assert not isinstance(ladder.omega, torch.nn.Parameter)
    assert {n for n, _ in ladder.named_buffers()} == {"omega"}      # no fitted statistic hides here


def test_omega_is_byte_identical_across_independently_built_ladders():
    """§6: two models built on different datasets must carry the same ω, bit for bit."""
    a, b = TimeLadder(), TimeLadder()
    assert torch.equal(a.omega, b.omega)
    assert a.omega.dtype == b.omega.dtype
    # and no shape anywhere depends on a dataset quantity (C, K, stype count) — only on n_freq
    assert a.omega.shape == (a.n_freq,)
    assert all(p.shape == () for p in a.parameters())


def test_load_state_dict_across_ladders_needs_no_dataset_agreement():
    """The cheapest proxy for LODO transfer: a ladder built anywhere loads into a ladder built anywhere."""
    src = TimeLadder()
    src.b_untimed.data.fill_(-1.25)
    dst = TimeLadder()
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert not missing and not unexpected
    assert float(dst.b_untimed.detach()) == -1.25
    assert torch.equal(dst.omega, src.omega)


def test_no_standardisation_hooks_exist():
    """§3.1/§0: no μ_τ, no σ_τ, no calibration slot may reappear on this module."""
    forbidden = {"mu_tau", "sigma_tau", "tau_mean", "tau_std", "mu", "sigma", "calibration", "scale"}
    ladder = TimeLadder()
    present = {n for n, _ in ladder.named_parameters()} | {n for n, _ in ladder.named_buffers()}
    assert not (present & forbidden), f"dataset-fitted statistic on the module: {present & forbidden}"


# ---- §9.10: the clamped-Delta flag ----


def test_was_clamped_detects_rows_dated_after_their_seed():
    """Must be computed from RAW times: after the clamp these are identical to a genuine Delta = 0."""
    ladder = TimeLadder()
    seed = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float64)
    row = torch.tensor([[50.0, 100.0, 150.0]], dtype=torch.float64)   # past, exact, FUTURE
    clamped = ladder.was_clamped(seed, row)
    assert clamped.tolist() == [[False, False, True]]
    # and the clamp really does erase the difference downstream
    tau = ladder.tau_from_times(seed, row)
    assert float(tau[0, 1]) == float(tau[0, 2]) == 0.0


def test_feats_separates_clamped_from_genuine_delta_zero():
    """The sinusoids cannot tell them apart; the indicator channel is the only thing that can."""
    ladder = TimeLadder()
    seed = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    row = torch.tensor([[100.0, 150.0]], dtype=torch.float64)         # Delta=0 exactly vs clamped
    timed = torch.ones(1, 2, dtype=torch.bool)
    tau = ladder.tau_from_times(seed, row)
    clamped = ladder.was_clamped(seed, row)

    f = ladder.feats(tau, timed, clamped)
    assert f.shape[-1] == ladder.feat_dim == 2 * ladder.n_freq + 1
    # the sinusoid block is identical...
    assert torch.allclose(f[0, 0, :-1], f[0, 1, :-1])
    # ...and only the flag differs
    assert float(f[0, 0, -1]) == 0.0 and float(f[0, 1, -1]) == 1.0


def test_feats_keeps_untimed_clamped_and_delta_zero_all_distinct():
    """Three states, three distinct encodings — untimed must not collide with either."""
    ladder = TimeLadder()
    seed = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float64)
    row = torch.tensor([[100.0, 150.0, 0.0]], dtype=torch.float64)
    timed = torch.tensor([[True, True, False]])
    clamped = ladder.was_clamped(seed, row) & timed
    f = ladder.feats(ladder.tau_from_times(seed, row), timed, clamped)

    delta0, clam, untimed = f[0, 0], f[0, 1], f[0, 2]
    assert not torch.allclose(delta0, clam)
    assert not torch.allclose(delta0, untimed)
    assert not torch.allclose(clam, untimed)
    assert bool((untimed == 0).all()), "untimed should be the all-zero encoding"


def test_clamped_bias_is_a_separate_learned_scalar_with_grad():
    ladder = TimeLadder()
    assert "b_clamped" in dict(ladder.named_parameters())
    q = torch.tensor([[False, True]])
    b = ladder.clamped_bias(q, q)
    assert b.shape == (1, 2, 2)
    with torch.no_grad():
        ladder.b_clamped.fill_(0.5)
    b = ladder.clamped_bias(q, q)
    # any pair touching the clamped endpoint picks up the bias; the (0,0) pair does not
    assert float(b[0, 0, 0]) == 0.0
    assert float(b[0, 0, 1]) == float(b[0, 1, 0]) == float(b[0, 1, 1]) == 0.5

    ladder.b_clamped.grad = None
    ladder.clamped_bias(q, q).sum().backward()
    assert ladder.b_clamped.grad is not None and float(ladder.b_clamped.grad) != 0.0


def test_time_bias_combines_untimed_and_clamped():
    ladder = TimeLadder()
    with torch.no_grad():
        ladder.b_untimed.fill_(0.25)
        ladder.b_clamped.fill_(0.5)
    timed = torch.tensor([[True, True]])
    clamped = torch.tensor([[False, True]])
    both = ladder.time_bias(timed, timed, clamped, clamped)
    assert float(both[0, 0, 0]) == 0.0                 # neither flag
    assert float(both[0, 1, 1]) == 0.5                 # clamped only
    # untimed-only path still works when no clamp info is supplied
    untimed = torch.tensor([[True, False]])
    assert float(ladder.time_bias(untimed, untimed)[0, 1, 1]) == 0.25
