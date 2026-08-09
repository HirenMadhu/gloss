"""FlexAttention backend for the cell level — the same attention, without the ``[B,S,S]`` bias.

**Why flex and not flash.** Cell attention carries an *additive* logit bias (§3.1 / §9.10): two
learnable scalars gated on per-token ``is_timed`` / ``was_clamped`` flags. flash-attn takes no
additive bias at all, so the current code folds the padding mask into the bias as ``-inf`` and hands
SDPA a dense float ``attn_mask``. That is the worst case for SDPA — it forces the math/mem-efficient
backend and materialises a ``[B,S,S]`` fp32 bias *plus* a ``[B,S,S]`` masked copy, per block, per
forward (≈1 GiB per block at ``B=512, S=512``). FlexAttention expresses the bias as a ``score_mod``
and the padding as a ``mask_mod``, so neither tensor is ever built.

**What it buys, honestly.** Padding in :mod:`gloss.data.collate` is a contiguous *suffix* (cells are
written at ``s = 0..n-1``), which is the best case for a ``BlockMask``: whole ``128×128`` blocks are
skipped rather than computed and discarded. But the win scales with the block grid, and at
``S=512`` that grid is only ``4×4``:

===========  ==========  ==================  ============================
``seq_len``  block grid  blocks kept (rel-f1/rel-event)  attention speedup
===========  ==========  ==================  ============================
512          4×4         ~4/16                ~4×
1536         12×12       ~25/144              ~5.8×
===========  ==========  ==================  ============================

Attention is only ~24% of per-block FLOPs, so at ``S=512`` this is ~1.2-1.3× end to end — real but
not the point. The point is that **512 → 1536 costs ~9× under SDPA and ~1.5× under flex**, which is
what makes a longer context affordable. amendments.md §1.3 measures rel-event's p90 cells/seed at
exactly the 512 cap, so that dataset is currently training at a silently reduced fanout.

**Two things to know before trusting it.**

* The bias scalars are *captured tensors* inside ``score_mod``. If autograd does not reach them they
  freeze silently and §9.10 is lost with no error — :func:`tests.test_flex_cell` asserts the
  gradient, and that assertion is the gate on this backend, not a nicety. Writing the capture the
  obvious way (a bare 0-dim parameter) compiles forward and then **fails to lower the backward**;
  :func:`cell_score_mod` carries the measured table of which forms work.
* flex's backward uses atomics, so ``--deterministic`` cannot stay bit-exact with this backend on.
  Determinism is the tool for measuring the noise floor; the two are not meant to be on together.

The row level is deliberately **not** converted: ``R ≤ 160`` is a ``2×2`` block grid, where block
sparsity buys nothing.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:  # torch >= 2.5; guarded so an older torch degrades to the SDPA backend instead of ImportError
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX = True
except Exception:  # pragma: no cover - depends on the installed torch
    create_block_mask = flex_attention = None  # type: ignore[assignment]
    HAS_FLEX = False

#: Flex's block granularity. Every ``seq_len`` we run (512/1024/1536) is a multiple of this, which
#: keeps the block grid exact; non-multiples still work, they just round up.
FLEX_BLOCK = 128

#: Flex lowers to ``tl.dot``, which cannot take an embedding dim below 16 — inductor raises
#: ``NotImplementedError`` from deep inside the lowering, at the first backward. The current grid
#: (``d_model=128, n_heads=8``) sits exactly on the limit, so this is a live constraint on the
#: shape sweep, not a theoretical one: 16 heads, or ``d_model=64``, would cross it.
FLEX_MIN_HEAD_DIM = 16

_COMPILED = None
_COMPILED_BUILDER = None


def _compiled_block_mask_builder():
    global _COMPILED_BUILDER
    if _COMPILED_BUILDER is None:
        _COMPILED_BUILDER = torch.compile(create_block_mask)
    return _COMPILED_BUILDER


def _flex_fn(device: torch.device):
    """``flex_attention``, compiled on CUDA and eager elsewhere.

    Compilation is what makes flex fast, but it needs a GPU backend to be worth anything and CPU
    eager flex is slow enough (minutes at ``S=128``) that compiling it would only add overhead to
    the tests. ``dynamic=False`` is deliberate: ``seq_len`` is a fixed cap, so shapes are static —
    except that the OOM backoff in ``run_gridsearch.py`` halves the batch, which costs one recompile.
    """
    global _COMPILED
    if device.type != "cuda":
        return flex_attention
    if _COMPILED is None:
        _COMPILED = torch.compile(flex_attention, dynamic=False)
    return _COMPILED


def build_cell_block_mask(is_padding: Tensor, *, block_size: int = FLEX_BLOCK):
    """``BlockMask`` for ``real[q] & real[k]`` — build **once per batch**, not once per block.

    Padding does not change between the blocks of a forward, and ``create_block_mask`` is the
    expensive part of the flex path, so the caller hoists this out of the block loop.
    """
    if not HAS_FLEX:
        raise RuntimeError("cell_attn_backend='flex' needs torch>=2.5 with flex_attention")
    real = ~is_padding                                        # [B,S]
    B, S = real.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return real[b, q_idx] & real[b, kv_idx]

    # `_compile=True` is deprecated in favour of compiling the builder itself; on CPU it is left
    # eager because the tests that run there are small and compiling would only add overhead.
    build = _compiled_block_mask_builder() if real.is_cuda else create_block_mask
    return build(mask_mod, B, None, S, S, device=real.device, BLOCK_SIZE=block_size)


def cell_score_mod(is_timed: Tensor, is_clamped: Tensor | None,
                   b_untimed: Tensor, b_clamped: Tensor | None,
                   *, dtype: torch.dtype | None = None):
    """The §3.1/§9.10 additive bias as a ``score_mod``.

    Mirrors ``TimeLadder.time_bias`` exactly: ``b_untimed·1[q or k untimed] + b_clamped·1[q or k
    clamped]``. Both flags are *per token*, which is why the dense ``[B,S,S]`` form was pure waste —
    it is an outer product of two length-``S`` boolean vectors, materialised in full.

    **Why the scalars are broadcast to ``[B,S]`` first.** Inductor places two undocumented
    restrictions on a captured tensor that requires grad, and the natural way to write this trips
    both. Measured on torch 2.8.0+cu128, forward compiles in every case; only the *backward* lowering
    differs, which is the dangerous kind of failure — see the note in ``tests/test_flex_cell.py``:

    ========================  ==================================================================
    capture form              backward
    ========================  ==================================================================
    0-dim scalar              ``LoweringException: AssertionError`` (flex_attention.py:2378)
    1-element 1-D             ``LoweringException: AssertionError``
    **``[B,S]`` broadcast**   **works — gradients reach both scalars**
    folded per-token terms    ``NotImplementedError``: a grad-carrying capture may be indexed once
    ========================  ==================================================================

    So each scalar is expanded to ``[B,S]`` outside the closure and indexed **exactly once** inside
    it. Every entry holds the same value, so the bias is unchanged; the gradient sums back over the
    broadcast, which is exactly what the scalar's gradient was. Cost is ``B×S`` floats (~1 MiB at
    ``512×512``) against the ~536 MiB ``[B,S,S]`` bias this replaces.

    ``is_timed`` / ``is_clamped`` are bool and carry no grad, so indexing them twice (once at ``q``,
    once at ``kv``) is fine — the restriction is only on grad-carrying captures.
    """
    B, S = is_timed.shape
    dt = dtype or b_untimed.dtype
    ub = b_untimed.to(dt).expand(B, S).contiguous()

    if is_clamped is None or b_clamped is None:
        def score_mod(score, b, h, q_idx, kv_idx):
            untimed = ~(is_timed[b, q_idx] & is_timed[b, kv_idx])
            return score + ub[b, q_idx] * untimed.to(score.dtype)
        return score_mod

    cb = b_clamped.to(dt).expand(B, S).contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        untimed = ~(is_timed[b, q_idx] & is_timed[b, kv_idx])
        clamped = is_clamped[b, q_idx] | is_clamped[b, kv_idx]
        return (score
                + ub[b, q_idx] * untimed.to(score.dtype)
                + cb[b, q_idx] * clamped.to(score.dtype))

    return score_mod


def flex_cell_attention(q: Tensor, k: Tensor, v: Tensor, *, block_mask, score_mod=None) -> Tensor:
    """``[B,H,S,d_h]`` in, same out. ``nan_to_num`` for the all-padding rows, as on the SDPA path."""
    out = _flex_fn(q.device)(q, k, v, score_mod=score_mod, block_mask=block_mask)
    return torch.nan_to_num(out)
