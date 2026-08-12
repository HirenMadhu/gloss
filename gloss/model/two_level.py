"""The two-level (cell, row) substrate — changes.md §3.2, §3.9.

Composes the row-level operators from :mod:`gloss.model.row_level` with the cell level into the
six-sublayer block of §3.9::

    1. cell attention        (temporal RoPE, padding-only mask)
    2. cell FFN              (MoEFFN, routed on the cell signature)
    3. low->high  RowPool
    4. row attention         (RoPE + name-derived role bias)
    5. row FFN               (RowMoE, shared + routed)
    6. high->low  Broadcast

**This is the adopted configuration, and the only one.** The phase-0a/0b ablation ladder (four RT
masks, no cell RoPE, bucket time, mean pooling, no row biases, dense row FFN) and the single-level
``arch: rt`` substrate are retired to ``archive/multi-level/``; every reported two-level result used
the settings hard-wired here, so the switches only ever had one live value.

**The MoE is at BOTH levels.** The cell FFN stays the existing ``MoEFFN`` routed on the cell
signature; the row FFN is a ``RowMoE`` routed on the row signature. Aux terms are returned *split by
level* so a collapse at one level cannot be hidden by the other.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from .flex_cell import (FLEX_MIN_HEAD_DIM, HAS_FLEX, build_cell_block_mask, cell_score_mod,
                        flex_cell_attention)
from .moe import MoEFFN, SwiGLU
from .row_level import Broadcast, RMSNorm, RowAttention, RowMoE, RowPool, RowSignature
from .time_encoding import TimeLadder

#: How the cell attention is *computed*. ``flex`` is the adopted backend — it is what makes
#: ``seq_len=3456`` on rel-event fit (19.8 GiB vs SDPA's 53.0 at B=64). ``sdpa`` is numerically
#: equivalent and stays as the reference the flex path is TESTED against: flex_attention needs CUDA
#: and ``head_dim >= 16``, so without sdpa there is no CPU test path at all.
CELL_BACKENDS = ("sdpa", "flex")


class CellAttention(nn.Module):
    """One full cell attention with a **padding-only** mask (§3.2), replacing the four masked ones.

    `col`, `feat`, `nbr` are deleted; `feat` and `nbr` are absorbed by the row level and `col` is
    dropped (Phase 0b's acceptance test is what licenses that — if it fails, the `col` mask was
    load-bearing and comes back as a third operator).

    Two things the measurements say about this, both recorded in amendments.md so the claim is not
    overstated:

    * Collapsing four attentions into one is exactly **4x** fewer score matrices *regardless* of mask
      density. The density figure supports a different claim.
    * The wasted-pair fraction is **84-96%** at ``seq_len=512`` depending on the dataset, not a single
      97-98%. Even collapsed, the remaining attention is only ~20% useful pairs on rel-f1, so varlen
      packing is where the rest of the win is.
    """

    def __init__(self, d_model: int, n_heads: int, ladder: TimeLadder, *, backend: str = "sdpa"):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if backend not in CELL_BACKENDS:
            raise ValueError(f"unknown cell_attn_backend {backend!r}; expected one of {CELL_BACKENDS}")
        if backend == "flex" and not HAS_FLEX:
            raise RuntimeError("cell_attn_backend='flex' needs torch>=2.5 with flex_attention")
        if backend == "flex" and d_model // n_heads < FLEX_MIN_HEAD_DIM:
            # `tl.dot` cannot take an embedding dim below 16, and inductor only says so ~40 lines
            # into a lowering dump, at the first backward of a training run. Say it here instead.
            # d_model=128/n_heads=8 sits exactly on the limit; halving d_model or doubling the heads
            # from the current grid would cross it.
            raise ValueError(
                f"cell_attn_backend='flex' needs d_model//n_heads >= {FLEX_MIN_HEAD_DIM}, got "
                f"{d_model}//{n_heads} = {d_model // n_heads}. Use fewer heads, a wider d_model, "
                f"or backend='sdpa'."
            )
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.backend = backend
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cb, block_mask=None) -> Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, S, self.h, self.d_h).transpose(1, 2)

        # temporal RoPE — always on (`cell.rope_time: false` was the phase-0 arm and is retired)
        tau = self.ladder.tau_from_times(cb.seed_time.unsqueeze(1), cb.row_time)
        theta = self.ladder.theta(tau, cb.is_timed)                  # [B,S,n_freq]
        th = theta.unsqueeze(1)
        q = self.ladder.rotate(q, th, self.n_rot)
        k = self.ladder.rotate(k, th, self.n_rot)
        # theta=0 alone reads as "Delta=0, maximally recent" for untimed cells AND for cells whose
        # Delta was clamped (§9.10); the additive flags are what separate the three states.
        clamped = self.ladder.was_clamped(cb.seed_time.unsqueeze(1), cb.row_time) & cb.is_timed

        if self.backend == "flex":
            # The bias is an outer product of two per-token boolean vectors, so flex expresses it as
            # a score_mod and never builds the [B,S,S] tensor the SDPA path below has to.
            if block_mask is None:
                block_mask = build_cell_block_mask(cb.is_padding)
            score_mod = cell_score_mod(cb.is_timed, clamped,
                                       self.ladder.b_untimed, self.ladder.b_clamped, dtype=x.dtype)
            out = flex_cell_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
            return self.wo(out.transpose(1, 2).reshape(B, S, -1))

        bias = self.ladder.time_bias(cb.is_timed, cb.is_timed,
                                     clamped, clamped).unsqueeze(1).to(x.dtype)
        real = ~cb.is_padding                                        # [B,S]
        mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(1)  # [B,1,S,S]
        # a float bias and a bool mask cannot both go to SDPA, so fold the mask into the bias
        attn = bias.masked_fill(~mask, float("-inf"))
        out = torch.nan_to_num(F.scaled_dot_product_attention(q, k, v, attn_mask=attn))
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))


class TwoLevelBlock(nn.Module):
    """The six-sublayer block of §3.9: cell attention, cell FFN, RowPool, row attention, row FFN,
    Broadcast. The phase-0a/0b ablation switches are retired — every switch is pinned to the `full`
    preset that every reported two-level result used."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_sig: int,
        ladder: TimeLadder,
        col_name_emb: Tensor,
        role_name_emb: Tensor,
        *,
        n_heads: int = 8,
        cell_attn_backend: str = "sdpa",
        cell_ffn: str = "moe",
        cell_route_dim: int | None = None,
        cell_num_experts: int = 4,
        cell_k: int = 2,
        cell_num_shared: int = 0,
        pool_slots: int = 4,
        row_num_experts: int = 4,
        row_k: int = 2,
        row_use_shared: bool = True,
        row_num_shared: int | None = None,
        lambda_ortho: float = 0.5,
        lambda_balance: float = 0.01,
        dispatch: str = "dense",
    ):
        super().__init__()
        self.cell_ffn_mode = cell_ffn

        # --- 1. cell attention (padding-only mask; RT's four masks are retired) ---
        self.cell_norm = RMSNorm(d_model)
        self.cell_attn = CellAttention(d_model, n_heads, ladder, backend=cell_attn_backend)

        # --- 2. cell FFN: the cell-level MoE (MoE at both levels) ---
        self.cell_ffn_norm = RMSNorm(d_model)
        if cell_ffn == "moe":
            self.cell_ffn = MoEFFN(d_model, d_ff, int(cell_route_dim or d_sig),
                                   num_experts=cell_num_experts, k=cell_k,
                                   num_shared=cell_num_shared, dispatch=dispatch)
        else:
            self.cell_ffn = SwiGLU(d_model, d_ff)

        # --- 3..6 row level ---
        self.pool = RowPool(d_model, d_sig, col_name_emb, slots=pool_slots)
        self.row_attn = RowAttention(d_model, d_sig, role_name_emb, ladder, n_heads=n_heads)
        self.row_ffn = RowMoE(d_model, d_ff, d_sig, num_experts=row_num_experts, k=row_k,
                              use_shared=row_use_shared, num_shared=row_num_shared,
                              lambda_ortho=lambda_ortho, lambda_balance=lambda_balance,
                              dispatch=dispatch)
        self.broadcast = Broadcast(d_model)

    def forward(self, h: Tensor, u: Tensor, z: Tensor, s: Tensor, cb, block_mask=None):
        """-> ``(h, u, aux_cell, aux_row, diag)``. Aux is split by LEVEL, deliberately."""
        h = h + self.cell_attn(self.cell_norm(h), cb, block_mask)          # 1. cell attention

        hn = self.cell_ffn_norm(h)                                         # 2. cell FFN
        if self.cell_ffn_mode == "moe":
            # Padding is handed to the sparse path and ignored by the dense one (see MoEFFN.forward).
            # It is the bigger of the two savings here: a sampled sequence runs 6-20% full at
            # seq_len=1024 on every DB measured except rel-f1's widest seed table.
            y, _g = self.cell_ffn(hn, z, valid=~cb.is_padding)
            h = h + y
            aux_cell = self.cell_ffn.ortho_loss()
        else:
            h = h + self.cell_ffn(hn)
            aux_cell = h.new_zeros(())

        u = self.pool(h, u, s, cb)                                         # 3. low -> high
        u, diag = self.row_attn(u, s, cb)                                  # 4. row attention
        u, aux_row, moe_diag = self.row_ffn(u, s, cb)                      # 5. row FFN
        diag = {**diag, **moe_diag}
        h = self.broadcast(h, u, cb)                                       # 6. high -> low
        return h, u, aux_cell, aux_row, diag


class TwoLevelSubstrate(nn.Module):
    """``n_blocks`` two-level blocks. ``forward -> (cell_states, row_states, aux, diag)``.

    Row tokens are initialised once, before block 0, as ``u_r^(0) = W_u[h̄_r ; s_r]`` (§3.4) — the
    mean of the row's own cell embeddings concatenated with its signature.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_sig: int,
        table_name_emb: Tensor,
        role_name_emb: Tensor,
        col_name_emb: Tensor,
        *,
        n_blocks: int = 6,
        n_heads: int = 8,
        max_hop: int = 8,
        ladder: TimeLadder | None = None,
        grad_checkpoint: bool = False,
        mean_aux: bool = False,
        **block_kw,
    ):
        super().__init__()
        self.ladder = ladder or TimeLadder()
        self.grad_checkpoint = grad_checkpoint
        # `aux` is a SUM over blocks, so its scale grows with depth AND with the expert count (the
        # ortho term is over an E x E Gram matrix). lambda_ortho=0.5 was tuned at E=4 / 8 blocks; at
        # E=8 / 10 blocks / two levels the same lambda is a different objective. `mean_aux` divides by
        # n_blocks so lambda means the same thing at any depth. Off by default: switching it on would
        # move every reported number.
        self.mean_aux = mean_aux
        self.row_sig = RowSignature(table_name_emb, role_name_emb, self.ladder,
                                    d_sig=d_sig, max_hop=max_hop)
        self.w_u = nn.Linear(d_model + d_sig, d_model, bias=False)
        self.blocks = nn.ModuleList(
            TwoLevelBlock(d_model, d_ff, d_sig, self.ladder, col_name_emb, role_name_emb,
                          n_heads=n_heads, **block_kw)
            for _ in range(n_blocks)
        )
        # Padding does not change between blocks, and `create_block_mask` is the expensive part of
        # the flex path — so it is built once per FORWARD here and handed to every block.
        self.needs_block_mask = block_kw.get("cell_attn_backend", "sdpa") == "flex"

    def _init_rows(self, h: Tensor, s: Tensor, cb) -> Tensor:
        B, R, _ = s.shape
        rows = torch.arange(R, device=h.device).view(1, R, 1)
        member = ((cb.cell_row.unsqueeze(1) == rows) & (~cb.is_padding).unsqueeze(1)).to(h.dtype)
        mean = (member @ h) / member.sum(-1, keepdim=True).clamp_min(1.0)
        return self.w_u(torch.cat([mean, s], dim=-1))

    def forward(self, x: Tensor, cb, *, z: Tensor | None = None):
        s = self.row_sig(cb)                          # [B,R,d_sig] — computed ONCE, reused per block
        u = self._init_rows(x, s, cb)
        block_mask = build_cell_block_mask(cb.is_padding) if self.needs_block_mask else None

        h = x
        aux_cell_total = x.new_zeros(())
        aux_row_total = x.new_zeros(())
        diags: list[dict] = []
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                # `use_reentrant=False` is required: the reentrant autograd path cannot handle a
                # block that returns a dict, and it silently drops grads for inputs that are not
                # tensors (z, s and cb all are, or contain, non-tensor state).
                h, u, aux_cell, aux_row, diag = torch.utils.checkpoint.checkpoint(
                    blk, h, u, z, s, cb, block_mask, use_reentrant=False)
            else:
                h, u, aux_cell, aux_row, diag = blk(h, u, z, s, cb, block_mask)
            aux_cell_total = aux_cell_total + aux_cell
            aux_row_total = aux_row_total + aux_row
            diags.append(diag)
        if self.mean_aux and self.blocks:
            aux_cell_total = aux_cell_total / len(self.blocks)
            aux_row_total = aux_row_total / len(self.blocks)
        aux = aux_cell_total + aux_row_total
        return h, u, aux, {"aux_cell": aux_cell_total, "aux_row": aux_row_total, "blocks": diags}
