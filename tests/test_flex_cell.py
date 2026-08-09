"""FlexAttention cell backend — equivalence with the SDPA path it replaces.

Split by what each check needs. The ``score_mod`` / ``mask_mod`` *definitions* are pure index
arithmetic and are checked exhaustively on CPU: that is where an off-by-one between ``q_idx`` and
``kv_idx`` would hide, and it costs nothing. The attention **output** and its **gradients** need a
GPU — CPU-eager flex takes minutes at ``S=128``, so those tests skip rather than crawl.

The load-bearing one is :func:`test_gradients_reach_the_captured_bias_scalars`. ``b_untimed`` and
``b_clamped`` are captured tensors inside a traced ``score_mod``; if autograd does not reach them
they simply stop training, with no error anywhere — the model would keep running and quietly lose
§9.10. Do not enable this backend on a machine where that test has not passed.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gloss.model.flex_cell import (HAS_FLEX, build_cell_block_mask, cell_score_mod,
                                   flex_cell_attention)
from gloss.model.time_encoding import TimeLadder
from gloss.model.two_level import CELL_BACKENDS, CellAttention, TwoLevelBlock

pytestmark = pytest.mark.skipif(not HAS_FLEX, reason="torch without flex_attention")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CPU-eager flex is unusably slow; this needs a GPU")


def make_flags(B=3, S=128, lens=(128, 71, 5), seed=0):
    """Padding as a contiguous suffix — what `collate.py` actually emits."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(S).unsqueeze(0)
    real = pos < torch.tensor(lens).unsqueeze(1)
    is_padding = ~real
    is_timed = real & (torch.rand(B, S, generator=g) > 0.3)
    is_clamped = is_timed & (torch.rand(B, S, generator=g) > 0.8)
    return is_padding, is_timed, is_clamped


# ------------------------------------------------------------------ mask_mod ---
def test_mask_mod_matches_the_dense_padding_mask():
    """The mask flex applies must be exactly the mask SDPA was given: `real[q] & real[k]`."""
    is_padding, _, _ = make_flags(B=2, S=64, lens=(64, 23))
    real = ~is_padding
    bm = build_cell_block_mask(is_padding, block_size=16)
    dense = real.unsqueeze(2) & real.unsqueeze(1)                    # [B,S,S]
    B, S = real.shape
    for b in range(B):
        for q in range(0, S, 7):                                    # stride: exhaustive is 4096 calls
            for kv in range(0, S, 5):
                got = bm.mask_mod(torch.tensor(b), torch.tensor(0),
                                  torch.tensor(q), torch.tensor(kv))
                assert bool(got) == bool(dense[b, q, kv]), (b, q, kv)


def test_padding_actually_removes_blocks():
    """The whole point: pad blocks are skipped, not computed and discarded."""
    S = 128
    padded, _, _ = make_flags(B=2, S=S, lens=(32, 32))
    full, _, _ = make_flags(B=2, S=S, lens=(S, S))
    assert build_cell_block_mask(padded, block_size=16).sparsity() > 50.0
    assert build_cell_block_mask(full, block_size=16).sparsity() == 0.0


def test_a_seed_with_no_real_cells_is_entirely_masked():
    is_padding = torch.ones(2, 32, dtype=torch.bool)
    is_padding[0, :16] = False
    bm = build_cell_block_mask(is_padding, block_size=16)
    assert not bool(bm.mask_mod(torch.tensor(1), torch.tensor(0),
                                torch.tensor(0), torch.tensor(0)))


# ----------------------------------------------------------------- score_mod ---
def test_score_mod_matches_time_bias_elementwise():
    """`score_mod` must reproduce `TimeLadder.time_bias` exactly, including the q/k asymmetry.

    `untimed` is an AND over the pair and `clamped` is an OR, so swapping the two operands or the
    two indices produces a bias that is wrong on exactly the mixed pairs — the ones that matter.
    """
    _, is_timed, is_clamped = make_flags(B=2, S=32, lens=(32, 19))
    ladder = TimeLadder()
    want = ladder.time_bias(is_timed, is_timed, is_clamped, is_clamped)   # [B,S,S]
    mod = cell_score_mod(is_timed, is_clamped, ladder.b_untimed, ladder.b_clamped)
    B, S = is_timed.shape
    base = torch.zeros(())
    for b in range(B):
        for q in range(0, S, 3):
            for kv in range(0, S, 3):
                got = mod(base, torch.tensor(b), torch.tensor(0),
                          torch.tensor(q), torch.tensor(kv))
                assert torch.allclose(got, want[b, q, kv], atol=1e-6), (b, q, kv)


def test_score_mod_without_the_clamp_flag_uses_only_the_untimed_term():
    _, is_timed, _ = make_flags(B=2, S=16, lens=(16, 9))
    ladder = TimeLadder()
    want = ladder.untimed_bias(is_timed, is_timed)
    mod = cell_score_mod(is_timed, None, ladder.b_untimed, None)
    for q in (0, 5, 15):
        for kv in (0, 7, 15):
            got = mod(torch.zeros(()), torch.tensor(0), torch.tensor(0),
                      torch.tensor(q), torch.tensor(kv))
            assert torch.allclose(got, want[0, q, kv], atol=1e-6)


def test_the_bias_capture_is_broadcast_not_a_bare_scalar():
    """A CPU guard on the one thing only a GPU can otherwise catch.

    Inductor refuses to lower a flex **backward** whose ``score_mod`` captures a 0-dim or 1-element
    grad-carrying tensor (``LoweringException: AssertionError``), while the forward compiles fine.
    Writing ``b_untimed`` in directly is therefore a change that looks correct, passes every forward
    test, and only explodes at ``.backward()`` — on a GPU, twelve minutes into a test run. This
    asserts the shape of what actually gets captured, so the mistake is caught in 2 seconds instead.
    """
    _, is_timed, is_clamped = make_flags(B=2, S=32, lens=(32, 19))
    ladder = TimeLadder()
    mod = cell_score_mod(is_timed, is_clamped, ladder.b_untimed, ladder.b_clamped)
    captured = [c.cell_contents for c in (mod.__closure__ or ())]
    grad_captures = [t for t in captured if torch.is_tensor(t) and t.requires_grad]
    assert len(grad_captures) == 2, "expected b_untimed and b_clamped to be captured with grad"
    for t in grad_captures:
        assert t.shape == is_timed.shape, f"capture is {tuple(t.shape)}, must be broadcast to [B,S]"
    # ...and it must still be the same number everywhere, or the bias is no longer a scalar bias
    for t, want in zip(grad_captures, (ladder.b_untimed, ladder.b_clamped)):
        assert torch.allclose(t, want.expand_as(t))


def test_score_mod_is_additive_on_the_incoming_score():
    """It modifies the logit, it does not replace it — a `return bias` bug would pass every test above."""
    _, is_timed, is_clamped = make_flags(B=1, S=8, lens=(8,))
    ladder = TimeLadder()
    mod = cell_score_mod(is_timed, is_clamped, ladder.b_untimed, ladder.b_clamped)
    args = (torch.tensor(0), torch.tensor(0), torch.tensor(3), torch.tensor(5))
    a = mod(torch.tensor(0.0), *args)
    b = mod(torch.tensor(2.5), *args)
    assert torch.allclose(b - a, torch.tensor(2.5), atol=1e-6)


# ------------------------------------------------------------------- wiring ---
def test_an_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="cell_attn_backend"):
        CellAttention(32, 4, TimeLadder(), backend="flash")


def test_flex_refuses_the_four_mask_parity_path():
    """`four_mask` is the frozen RT parity guard; a different kernel there would void the guard."""
    with pytest.raises(ValueError, match="requires cell_attention='full'"):
        TwoLevelBlock(32, 64, 16, TimeLadder(), torch.randn(5, 12), torch.randn(4, 12),
                      n_heads=4, cell_attention="four_mask", cell_attn_backend="flex")


def test_a_head_dim_below_16_is_rejected_at_construction():
    """Flex lowers to `tl.dot`, which needs head_dim >= 16 — and only says so ~40 lines into an
    inductor dump, at the first backward of a training run. `d_model=128, n_heads=8` is exactly on
    the limit, so this is one grid step away from being hit for real."""
    with pytest.raises(ValueError, match="d_model//n_heads >= 16"):
        CellAttention(32, 4, TimeLadder(), backend="flex")
    CellAttention(32, 2, TimeLadder(), backend="flex")          # d_h == 16 is allowed
    CellAttention(32, 4, TimeLadder(), backend="sdpa")          # sdpa has no such limit


def test_sdpa_is_the_default():
    """Every result before this change used SDPA; the backend must not switch itself on."""
    assert CELL_BACKENDS[0] == "sdpa"
    assert CellAttention(32, 4, TimeLadder()).backend == "sdpa"


def test_the_substrate_builds_the_block_mask_once_per_forward():
    """Per-block construction would pay `create_block_mask` n_blocks times for an identical mask."""
    from gloss.model.two_level import TwoLevelSubstrate

    kw = dict(d_sig=16, n_blocks=2, n_heads=4, table_name_emb=torch.randn(3, 12),
              role_name_emb=torch.randn(4, 12), col_name_emb=torch.randn(5, 12))
    sub = TwoLevelSubstrate(32, 64, kw["d_sig"], kw["table_name_emb"], kw["role_name_emb"],
                            kw["col_name_emb"], n_blocks=2, n_heads=2,
                            cell_attention="full", cell_attn_backend="flex")
    assert sub.needs_block_mask
    plain = TwoLevelSubstrate(32, 64, kw["d_sig"], kw["table_name_emb"], kw["role_name_emb"],
                              kw["col_name_emb"], n_blocks=2, n_heads=2, cell_attention="full")
    assert not plain.needs_block_mask


# --------------------------------------------------------------- GPU: output ---
def _sdpa_reference(q, k, v, is_padding, is_timed, is_clamped, ladder):
    bias = ladder.time_bias(is_timed, is_timed, is_clamped, is_clamped).unsqueeze(1).to(q.dtype)
    real = ~is_padding
    mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(1)
    return torch.nan_to_num(
        F.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(~mask, float("-inf"))))


@needs_cuda
def test_flex_output_matches_sdpa():
    dev = torch.device("cuda")
    B, H, S, D = 2, 4, 256, 32
    is_padding, is_timed, is_clamped = make_flags(B=B, S=S, lens=(256, 137))
    is_padding, is_timed, is_clamped = (t.to(dev) for t in (is_padding, is_timed, is_clamped))
    q, k, v = (torch.randn(B, H, S, D, device=dev) for _ in range(3))
    ladder = TimeLadder().to(dev)

    got = flex_cell_attention(q, k, v, block_mask=build_cell_block_mask(is_padding),
                              score_mod=cell_score_mod(is_timed, is_clamped,
                                                       ladder.b_untimed, ladder.b_clamped))
    want = _sdpa_reference(q, k, v, is_padding, is_timed, is_clamped, ladder)
    # only the real rows are compared: a fully-masked query row is 0 by convention on both paths,
    # and asserting on it would test `nan_to_num`, not the attention.
    real = (~is_padding).unsqueeze(1).unsqueeze(-1).expand_as(got)
    assert torch.allclose(got[real], want[real], atol=2e-3, rtol=2e-3), \
        (got[real] - want[real]).abs().max()


@needs_cuda
def test_all_padding_rows_are_zero_not_nan():
    dev = torch.device("cuda")
    is_padding = torch.ones(2, 128, dtype=torch.bool, device=dev)
    is_padding[0, :64] = False
    q, k, v = (torch.randn(2, 4, 128, 32, device=dev) for _ in range(3))
    out = flex_cell_attention(q, k, v, block_mask=build_cell_block_mask(is_padding))
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[1]) == 0


@needs_cuda
def test_the_whole_substrate_agrees_across_backends():
    """End to end, not just the attention op.

    The unit tests above would all pass with the block mask never reaching the blocks (each
    ``CellAttention`` would quietly rebuild its own). This is the test that pins the plumbing:
    identical weights, identical batch, two backends, compared on the output *and* on the gradient
    of every parameter.
    """
    import types

    from gloss.model.two_level import TwoLevelSubstrate

    from .test_row_level import D_MODEL, D_SIG, _tables
    from .test_row_level import stub_batch

    dev = torch.device("cuda")
    cb, K = stub_batch(B=2, R=16, S=256, K=3)
    cb = types.SimpleNamespace(**{k: (v.to(dev) if torch.is_tensor(v) else v)
                                  for k, v in vars(cb).items()})
    tab, role, col = _tables(K)

    def build(backend):
        torch.manual_seed(0)                        # same init for both, or the comparison is noise
        # n_heads=2, not the stub's usual 4: flex needs d_model//n_heads >= FLEX_MIN_HEAD_DIM
        return TwoLevelSubstrate(D_MODEL, 2 * D_MODEL, D_SIG, tab, role, col,
                                 n_blocks=2, n_heads=2, cell_attention="full",
                                 cell_rope_time=True, time_bias="rope",
                                 cell_attn_backend=backend).to(dev)

    torch.manual_seed(1)
    x = torch.randn(cb.num_seeds, cb.seq_len, D_MODEL, device=dev)
    z = torch.randn(cb.num_seeds, cb.seq_len, D_SIG, device=dev)

    def run(backend):
        sub = build(backend)
        h, u, aux, _ = sub(x, cb, z=z)
        (h.square().sum() + u.square().sum() + aux).backward()
        return h, u, {n: p.grad for n, p in sub.named_parameters() if p.grad is not None}

    h_f, u_f, g_f = run("flex")
    h_s, u_s, g_s = run("sdpa")
    assert torch.allclose(h_f, h_s, atol=2e-3, rtol=2e-3), (h_f - h_s).abs().max()
    assert torch.allclose(u_f, u_s, atol=2e-3, rtol=2e-3), (u_f - u_s).abs().max()
    assert set(g_f) == set(g_s)
    for n in g_f:
        assert torch.allclose(g_f[n], g_s[n], atol=2e-2, rtol=2e-2), \
            f"{n}: max diff {(g_f[n] - g_s[n]).abs().max().item():.3e}"
    assert "ladder.b_untimed" in g_f or any("b_untimed" in n for n in g_f), \
        "the time-bias scalars got no gradient through the substrate"


@needs_cuda
def test_gradients_reach_the_captured_bias_scalars():
    """THE gate on this backend.

    `b_untimed` / `b_clamped` are captured by the traced `score_mod`. If flex does not backprop into
    captured tensors, they freeze at init and §9.10 silently stops being learned — no exception, no
    NaN, just a model that quietly lost a mechanism. Compare against SDPA rather than merely asserting
    non-None, because a gradient that exists but is wrong is the worse failure.
    """
    dev = torch.device("cuda")
    B, H, S, D = 2, 4, 128, 32
    is_padding, is_timed, is_clamped = make_flags(B=B, S=S, lens=(128, 61))
    is_padding, is_timed, is_clamped = (t.to(dev) for t in (is_padding, is_timed, is_clamped))
    q, k, v = (torch.randn(B, H, S, D, device=dev, requires_grad=True) for _ in range(3))
    ladder = TimeLadder().to(dev)
    for p in (ladder.b_untimed, ladder.b_clamped):
        assert p.requires_grad, "the bias scalars are not learnable; this test proves nothing"

    def grads(out):
        return torch.autograd.grad(out.square().sum(),
                                   [q, k, v, ladder.b_untimed, ladder.b_clamped],
                                   allow_unused=True)

    g_flex = grads(flex_cell_attention(
        q, k, v, block_mask=build_cell_block_mask(is_padding),
        score_mod=cell_score_mod(is_timed, is_clamped, ladder.b_untimed, ladder.b_clamped)))
    g_ref = grads(_sdpa_reference(q, k, v, is_padding, is_timed, is_clamped, ladder))

    for name, a, b in zip(["q", "k", "v", "b_untimed", "b_clamped"], g_flex, g_ref):
        assert a is not None, f"flex produced NO gradient for {name}"
        assert torch.isfinite(a).all(), name
        assert torch.allclose(a, b, atol=2e-2, rtol=2e-2), \
            f"{name}: max diff {(a - b).abs().max().item():.3e}"
