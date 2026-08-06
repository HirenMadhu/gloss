"""The two-level (cell, row) substrate — changes.md §3.2, §3.9.

Composes the row-level operators from :mod:`gloss.model.row_level` with the cell level into the
six-sublayer block of §3.9::

    1. cell attention        (temporal RoPE, pad mask)   -- or the four RT masks, for Phase 0a
    2. cell FFN              (MoEFFN, unchanged)
    3. low->high  RowPool
    4. row attention         (RoPE + name-derived role bias)
    5. row FFN               (RowMoE)
    6. high->low  Broadcast

``rt_substrate.py`` is deliberately left untouched and reachable, so ``arch: rt`` can A/B against
this and the §6 parity guard keeps working. Every stage here is switchable, because each phase in §5
must be independently ablatable — Phase 0a runs this module with the cell level configured to behave
exactly like RT (``cell.attention: four_mask``, ``cell.rope_time: false``), which is what isolates
the row-token addition.

**The MoE is at BOTH levels.** The cell FFN stays the existing ``MoEFFN`` routed on the cell
signature; the row FFN is a ``RowMoE`` routed on the row signature. Aux terms are returned *split by
level* so a collapse at one level cannot be hidden by the other.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .moe import MoEFFN, SwiGLU
from .row_level import Broadcast, RMSNorm, RowAttention, RowMoE, RowPool, RowSignature
from .rt_substrate import REL_ORDER, MaskedAttention, build_relational_masks
from .time_encoding import TimeLadder


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

    def __init__(self, d_model: int, n_heads: int, ladder: TimeLadder, *, rope_time: bool = True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.rope_time = rope_time
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cb) -> Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, S, self.h, self.d_h).transpose(1, 2)

        bias = None
        if self.rope_time:
            tau = self.ladder.tau_from_times(cb.seed_time.unsqueeze(1), cb.row_time)
            theta = self.ladder.theta(tau, cb.is_timed)              # [B,S,n_freq]
            th = theta.unsqueeze(1)
            q = self.ladder.rotate(q, th, self.n_rot)
            k = self.ladder.rotate(k, th, self.n_rot)
            # theta=0 alone reads as "Delta=0, maximally recent" for untimed cells AND for cells
            # whose Delta was clamped (§9.10); the additive flags are what separate the three states.
            clamped = self.ladder.was_clamped(cb.seed_time.unsqueeze(1), cb.row_time) & cb.is_timed
            bias = self.ladder.time_bias(cb.is_timed, cb.is_timed,
                                         clamped, clamped).unsqueeze(1).to(x.dtype)

        real = ~cb.is_padding                                        # [B,S]
        mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(1)  # [B,1,S,S]
        if bias is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # a float bias and a bool mask cannot both go to SDPA, so fold the mask into the bias
            attn = bias.masked_fill(~mask, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn)
        out = torch.nan_to_num(out)                                  # all-pad rows -> 0, not NaN
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))


class TwoLevelBlock(nn.Module):
    """The six-sublayer block of §3.9. Every stage switchable so each phase is ablatable alone."""

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
        cell_attention: str = "full",
        cell_rope_time: bool = True,
        cell_ffn: str = "moe",
        cell_route_dim: int | None = None,
        cell_num_experts: int = 4,
        cell_k: int = 2,
        pool_query: str = "hybrid",
        pool_slots: int = 4,
        role_bias: str = "name_derived",
        time_bias: str = "rope",
        row_ffn: str = "moe",
        row_num_experts: int = 4,
        row_k: int = 2,
        row_use_shared: bool = True,
        lambda_ortho: float = 0.5,
        lambda_balance: float = 0.01,
        broadcast: str = "additive",
    ):
        super().__init__()
        if cell_attention not in ("full", "four_mask"):
            raise ValueError(f"unknown cell.attention {cell_attention!r}")
        self.cell_attention = cell_attention
        self.cell_ffn_mode = cell_ffn
        self.row_ffn_mode = row_ffn

        # --- 1. cell attention ---
        self.cell_norm = RMSNorm(d_model)
        if cell_attention == "full":
            self.cell_attn = CellAttention(d_model, n_heads, ladder, rope_time=cell_rope_time)
        else:
            # Phase 0a: keep RT's four masked attentions verbatim, so the only change is row tokens
            self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in REL_ORDER})
            self.attns = nn.ModuleDict({l: MaskedAttention(d_model, n_heads) for l in REL_ORDER})

        # --- 2. cell FFN: the EXISTING cell-level MoE, unchanged (MoE at both levels) ---
        self.cell_ffn_norm = RMSNorm(d_model)
        if cell_ffn == "moe":
            self.cell_ffn = MoEFFN(d_model, d_ff, int(cell_route_dim or d_sig),
                                   num_experts=cell_num_experts, k=cell_k)
        else:
            self.cell_ffn = SwiGLU(d_model, d_ff)

        # --- 3..6 row level ---
        self.pool = RowPool(d_model, d_sig, col_name_emb, slots=pool_slots, mode=pool_query)
        self.row_attn = RowAttention(d_model, d_sig, role_name_emb, ladder, n_heads=n_heads,
                                     role_bias=role_bias, time_bias=time_bias)
        if row_ffn == "moe":
            self.row_ffn = RowMoE(d_model, d_ff, d_sig, num_experts=row_num_experts, k=row_k,
                                  use_shared=row_use_shared,
                                  lambda_ortho=lambda_ortho, lambda_balance=lambda_balance)
        else:
            self.row_ffn_norm = RMSNorm(d_model)
            self.row_ffn = SwiGLU(d_model, d_ff)
        self.broadcast = Broadcast(d_model, mode=broadcast)

    def forward(self, h: Tensor, u: Tensor, z: Tensor, s: Tensor, cb, masks=None):
        """-> ``(h, u, aux_cell, aux_row, diag)``. Aux is split by LEVEL, deliberately."""
        # 1. cell attention
        if self.cell_attention == "full":
            h = h + self.cell_attn(self.cell_norm(h), cb)
        else:
            for l in REL_ORDER:
                h = h + self.attns[l](self.norms[l](h), masks[l])

        # 2. cell FFN
        hn = self.cell_ffn_norm(h)
        if self.cell_ffn_mode == "moe":
            y, _g = self.cell_ffn(hn, z)
            h = h + y
            aux_cell = self.cell_ffn.ortho_loss()
        else:
            h = h + self.cell_ffn(hn)
            aux_cell = h.new_zeros(())

        # 3. low -> high
        u = self.pool(h, u, s, cb)

        # 4. row attention
        u, diag = self.row_attn(u, s, cb)

        # 5. row FFN
        if self.row_ffn_mode == "moe":
            u, aux_row, moe_diag = self.row_ffn(u, s, cb)
            diag = {**diag, **moe_diag}
        else:
            u = u + self.row_ffn(self.row_ffn_norm(u))
            aux_row = u.new_zeros(())

        # 6. high -> low
        h = self.broadcast(h, u, cb)
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
        recency_channel: str = "off",
        **block_kw,
    ):
        super().__init__()
        self.ladder = ladder or TimeLadder()
        self.row_sig = RowSignature(table_name_emb, role_name_emb, self.ladder,
                                    d_sig=d_sig, max_hop=max_hop)
        self.w_u = nn.Linear(d_model + d_sig, d_model, bias=False)
        # The recency order-statistic channel reads the *truncation window* of each role's sampled
        # child set. It is injected into the row token ONCE, before block 0's row attention, and only
        # ever into the value path — `s` (what both routers read) is built above and never sees it.
        self.recency_channel = recency_channel
        if recency_channel != "off":
            from .recency_stats import RecencyOrderChannel

            # Built inside a FORKED RNG so constructing it does not advance the global stream. Without
            # this, `x_full` and `base` at the same seed get different weights for every parameter
            # created after this point, and the arms stop being paired -- which would quietly turn an
            # init difference into "the mechanism helped".
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.x_channel = RecencyOrderChannel(d_model, role_name_emb, mode=recency_channel)
        self.blocks = nn.ModuleList(
            TwoLevelBlock(d_model, d_ff, d_sig, self.ladder, col_name_emb, role_name_emb,
                          n_heads=n_heads, **block_kw)
            for _ in range(n_blocks)
        )
        self.needs_masks = block_kw.get("cell_attention", "full") == "four_mask"

    def _init_rows(self, h: Tensor, s: Tensor, cb) -> Tensor:
        B, R, _ = s.shape
        rows = torch.arange(R, device=h.device).view(1, R, 1)
        member = ((cb.cell_row.unsqueeze(1) == rows) & (~cb.is_padding).unsqueeze(1)).to(h.dtype)
        mean = (member @ h) / member.sum(-1, keepdim=True).clamp_min(1.0)
        return self.w_u(torch.cat([mean, s], dim=-1))

    def forward(self, x: Tensor, cb, *, z: Tensor | None = None):
        s = self.row_sig(cb)                          # [B,R,d_sig] — computed ONCE, reused per block
        u = self._init_rows(x, s, cb)
        x_diag: dict = {}
        if self.recency_channel != "off":
            h_x, x_diag = self.x_channel(cb)          # already scaled by alpha (init 0)
            u = u + h_x
        masks = build_relational_masks(cb) if self.needs_masks else None

        h = x
        aux = x.new_zeros(())
        aux_cell_total = x.new_zeros(())
        aux_row_total = x.new_zeros(())
        diags: list[dict] = []
        for blk in self.blocks:
            h, u, aux_cell, aux_row, diag = blk(h, u, z, s, cb, masks)
            aux_cell_total = aux_cell_total + aux_cell
            aux_row_total = aux_row_total + aux_row
            diags.append(diag)
        aux = aux_cell_total + aux_row_total
        return h, u, aux, {"aux_cell": aux_cell_total, "aux_row": aux_row_total, "blocks": diags,
                           **x_diag}
