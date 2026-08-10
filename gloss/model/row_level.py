"""Row-level modules for the two-level encoder — changes.md §3.3–§3.7.

Rows currently have no representation in the RT substrate; the four boolean masks are the *only*
encoding of row structure. This module adds a row token `u_r` per sampled row and the four operators
that read and write it:

* :class:`RowSignature` (§3.3) — value-free, name-derived `s_r`, computed once per forward.
* :class:`RowPool`      (§3.4) — low→high cross-attention, cells → row token.
* :class:`RowAttention` (§3.5) — row↔row attention over `adj_role`, temporal RoPE + name-derived γ.
* :class:`Broadcast`    (§3.6) — high→low, row token back onto its cells.
* :class:`RowMoE`       (§3.7) — cosine-routed MoE on `s_r`, **added alongside** the existing
  cell-level `MoEFFN`, not substituted for it.

**The §0 no-dataset-artifact rule governs every parameter here.** `K` (role count) and `n_tables` are
new on an unseen schema, so nothing may be *indexed* by them. The frozen name tables are registered
``persistent=False`` precisely so no `state_dict` entry has a K- or n_tables-shaped row count, and γ
is a bilinear form on the frozen role-name embedding rather than a `R^{2K+2}` lookup. That is what
makes a checkpoint loadable on a schema it never trained on.

Time enters only through :class:`~gloss.model.time_encoding.TimeLadder` — fixed frequencies, never
learned. (`archive/halos/time_encoding.py` makes ω an `nn.Parameter`; that is the artifact §0 forbids,
so do not copy from it.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .moe import SwiGLU
from .time_encoding import TimeLadder

# `adj_role` direction classes for γ's `c^(h)_dir` term (§3.5): three learned directions, so `K`
# never sizes a parameter.
DIR_CHILD, DIR_PARENT, DIR_SELF = 0, 1, 2
N_DIR = 3


class RMSNorm(nn.Module):
    """Matches the substrate's pre-norm convention."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _frozen(module: nn.Module, name: str, t: Tensor) -> None:
    """Register a frozen name table as a NON-persistent buffer.

    Non-persistent is load-bearing, not tidiness: these tables are `[n_tables, d_text]` and
    `[K, d_text]`, so a *persistent* buffer would put a training-set-sized shape into `state_dict`
    and break §6's no-dataset-artifact guard — and a checkpoint would refuse to load on a schema
    with a different number of tables or roles.
    """
    module.register_buffer(name, t.detach().to(torch.float32), persistent=False)


class RowSignature(nn.Module):
    r"""§3.3 — the value-free row signature, shared by the row encoder query and the row router.

    .. math::
        s_r = \mathrm{RMSNorm}\big(W_{tab}\,\mathrm{name}_{k(r)} + W_\rho\,\mathrm{name}_{\rho_{in}(r)}
              + W_\eta\,\mathrm{emb}(\eta_r) + W_\tau[\sin\theta_r;\cos\theta_r]\big)

    Every term is name-derived, a small universal integer (`hop`), or a fixed basis — so `s_r` is
    defined on an unseen schema. Computed **once** per forward and reused by every block.

    `role_name_emb` must be the ``[K+1, d_text]`` variant whose row 0 is the all-zero ``FK_NONE``
    slot (``schema.role_name_embeddings_with_none``), so ``row_in_role``'s 0 for the root gathers a
    zero vector without an offset.
    """

    def __init__(
        self,
        table_name_emb: Tensor,
        role_name_emb: Tensor,
        ladder: TimeLadder,
        *,
        d_sig: int = 128,
        max_hop: int = 8,
    ):
        super().__init__()
        _frozen(self, "table_name_emb", table_name_emb)
        _frozen(self, "role_name_emb", role_name_emb)
        self.ladder = ladder
        self.d_sig = d_sig
        d_text = table_name_emb.shape[-1]
        self.w_tab = nn.Linear(d_text, d_sig, bias=False)
        self.w_rho = nn.Linear(role_name_emb.shape[-1], d_sig, bias=False)
        # hop is a small universal integer, so an embedding table over it is legal under §0
        self.hop_emb = nn.Embedding(max_hop + 1, d_sig)
        self.w_tau = nn.Linear(ladder.feat_dim, d_sig, bias=False)
        self.norm = RMSNorm(d_sig)
        self.max_hop = max_hop

    def forward(self, cb) -> Tensor:
        """-> ``[B, R, d_sig]``. Padding rows get a well-defined (unused) signature, never NaN."""
        table = cb.row_table.clamp_min(0)                     # pad is -1; clamp so gather is safe
        role = cb.row_in_role.clamp(0, self.role_name_emb.shape[0] - 1)
        hop = cb.row_hop.clamp(0, self.max_hop)

        z = self.w_tab(self.table_name_emb[table])
        z = z + self.w_rho(self.role_name_emb[role])
        z = z + self.hop_emb(hop)

        seed_t = cb.seed_time.unsqueeze(1)
        tau = self.ladder.tau_from_times(seed_t, cb.row_time_r)
        # §9.10: rows dated AFTER their seed had Delta clamped to 0, so their sinusoids are identical
        # to a genuine Delta = 0. The indicator channel is the only thing that separates them.
        clamped = self.ladder.was_clamped(seed_t, cb.row_time_r) & cb.row_is_timed
        z = z + self.w_tau(self.ladder.feats(tau, cb.row_is_timed, clamped).to(z.dtype))
        return self.norm(z)


class RowPool(nn.Module):
    r"""§3.4 — low→high. Cross-attention from a row's own cells into its row token.

    Keys are **column-name** embeddings, values are **cell states** (Griffin's split), so the
    attention decides *which columns of this row matter* rather than being driven by cell content.
    `M` query slots, concatenated then projected.

    The softmax is scoped to `{i : ν(i) = r}` — a row attends only over its OWN cells. Cells of
    other rows and padding cells are masked out, so a row with no cells yields zero rather than NaN.

    `mode` deliberately mirrors the existing `route_on` vocabulary:
    ``mean`` (no cross-attention) | ``signature`` (q = W_Q s_r) | ``hidden`` (q = W_Q u_r) |
    ``hybrid`` (q = W_Q[u_r ; s_r]).
    """

    def __init__(
        self,
        d_model: int,
        d_sig: int,
        col_name_emb: Tensor,
        *,
        slots: int = 4,
        mode: str = "hybrid",
        d_head: int | None = None,
    ):
        super().__init__()
        if mode not in ("mean", "signature", "hidden", "hybrid"):
            raise ValueError(f"unknown pool_query mode {mode!r}")
        _frozen(self, "col_name_emb", col_name_emb)
        self.mode = mode
        self.slots = 1 if mode == "mean" else slots
        self.d_h = d_head or d_model // 4
        if mode != "mean":
            d_q = {"signature": d_sig, "hidden": d_model, "hybrid": d_model + d_sig}[mode]
            self.w_q = nn.Linear(d_q, self.slots * self.d_h, bias=False)
            self.w_k = nn.Linear(col_name_emb.shape[-1], self.d_h, bias=False)
            self.w_v = nn.Linear(d_model, self.d_h, bias=False)
            self.w_o = nn.Linear(self.slots * self.d_h, d_model, bias=False)

    def _membership(self, cb, R: int) -> Tensor:
        """``[B, R, S]`` bool — cell `i` belongs to row `r` and is not padding.

        Kept only as the reference the segment-softmax path in :meth:`forward` is tested against; it
        is deliberately *not* used at runtime, because materializing it is the whole problem below.
        """
        rows = torch.arange(R, device=cb.cell_row.device).view(1, R, 1)
        return (cb.cell_row.unsqueeze(1) == rows) & (~cb.is_padding).unsqueeze(1)

    def _groups(self, cb, R: int) -> tuple[Tensor, Tensor]:
        """``(valid [B,S], flat_group_id [B*S])`` — the row each cell belongs to, flattened over B.

        Cells past `MAX_ROWS` truncation (``row >= R``) and padding cells (``node_idxs == -1``) are
        dropped, matching what ``_membership`` silently did by never matching them.
        """
        row_of = cb.cell_row                                              # [B,S]; pad = -1
        valid = (~cb.is_padding) & (row_of >= 0) & (row_of < R)
        idx = row_of.clamp(0, R - 1)                                      # safe index; masked below
        base = torch.arange(cb.cell_row.shape[0], device=row_of.device).unsqueeze(1) * R
        return valid, (base + idx).reshape(-1)

    def forward(self, h: Tensor, u: Tensor, s: Tensor, cb) -> Tensor:
        """``h [B,S,d]``, ``u [B,R,d]``, ``s [B,R,d_sig]`` -> updated ``u [B,R,d]`` (residual).

        Implemented as a **segment softmax over cells grouped by row**, not as dense ``[B,R,M,S]``
        attention. Every cell belongs to exactly one row, so of a dense score tensor precisely `1/R`
        of the entries survive the membership mask — at B=256, R=160, M=4, S=512 that is 84M scores
        (~336 MB fp32, kept for backward, twice) of which 99.4% are `-inf`. The grouped form computes
        the same softmax over the same sets in ``[B,S,M]``, so it is exact rather than an
        approximation; `test_row_pool.py` pins it against `_membership` above.

        One honest cost: the two `index_add` reductions use CUDA atomics, so this path is
        run-to-run non-deterministic in the last bits where the einsum was not. That is covered by
        `seed_everything(..., deterministic_torch=True)`, which has deterministic implementations for
        both.
        """
        B, R, _ = u.shape
        S = h.shape[1]
        M, dh = self.slots, self.d_h
        n_groups = B * R
        valid, flat_g = self._groups(cb, R)                               # [B,S], [B*S]

        if self.mode == "mean":
            d = h.shape[-1]
            w = valid.to(h.dtype).unsqueeze(-1)
            acc = h.new_zeros(n_groups, d).index_add(0, flat_g, (h * w).reshape(-1, d))
            cnt = h.new_zeros(n_groups).index_add(0, flat_g, w.reshape(-1)).clamp_min(1)
            return u + (acc / cnt.unsqueeze(-1)).view(B, R, d)

        q_in = {"signature": s, "hidden": u, "hybrid": torch.cat([u, s], dim=-1)}[self.mode]
        q = self.w_q(q_in).view(B, R, M, dh)                              # [B,R,M,dh]
        k = self.w_k(self.col_name_emb[cb.col_idxs.clamp_min(0)])         # [B,S,dh]
        v = self.w_v(h)                                                   # [B,S,dh]

        # Each cell scores against ITS OWN row's queries only, so gather q by row rather than
        # broadcasting it over all R.
        idx = flat_g.view(B, S) - torch.arange(B, device=h.device).unsqueeze(1) * R
        q_cell = q.gather(1, idx.view(B, S, 1, 1).expand(B, S, M, dh))    # [B,S,M,dh]
        scores = torch.matmul(q_cell.reshape(B * S, M, dh),
                              k.reshape(B * S, dh, 1)).view(B, S, M) / dh ** 0.5
        # finfo.min rather than -inf: an all-masked group would otherwise give (-inf) - (-inf) = NaN
        # in the shift below, and NaN survives the zeroing that a finite sentinel does not need.
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid.unsqueeze(-1), neg)

        gm = scores.new_full((n_groups, M), neg).scatter_reduce(
            0, flat_g.unsqueeze(-1).expand(-1, M), scores.detach().reshape(-1, M),
            reduce="amax", include_self=True)
        e = torch.exp(scores - gm.index_select(0, flat_g).view(B, S, M))
        e = torch.where(valid.unsqueeze(-1), e, torch.zeros_like(e))
        den = e.new_zeros(n_groups, M).index_add(0, flat_g, e.reshape(-1, M))
        # a row with no cells has den == 0 and no valid cell pointing at it -> its pooled value stays
        # exactly 0, which is what the dense path's explicit `empty` zeroing did.
        a = e / den.index_select(0, flat_g).view(B, S, M).clamp_min(torch.finfo(e.dtype).tiny)

        contrib = torch.matmul(a.reshape(B * S, M, 1), v.reshape(B * S, 1, dh))   # [B*S,M,dh]
        pooled = contrib.new_zeros(n_groups, M, dh).index_add(0, flat_g, contrib)
        return u + self.w_o(pooled.view(B, R, M * dh))


class RowAttention(nn.Module):
    r"""§3.5 — row↔row attention over the sampled row graph.

    .. math::
        s^{(h)}_{rs} = \frac{\tilde q_r^{(h)\top}\tilde k_s^{(h)}}{\sqrt{d_h}}
                       + \gamma^{(h)}_{rs} + b_{untimed}\mathbb{1}[r\ \text{or}\ s\ \text{untimed}]

    with `q̃ = R(θ)W_q u` — temporal RoPE, the same ladder as the cell level. Mask is
    ``adj_role != 0`` AND-ed with ``row_valid`` on both axes; self-loops are always present so no
    row is ever fully masked.

    **γ is name-derived** (`role_bias='name_derived'`):

    .. math::
        \gamma^{(h)}_{rs} = \langle v^{(h)}, W_\rho\,\mathrm{name}_{\rho(r,s)}\rangle + c^{(h)}_{dir(r,s)}

    An earlier draft used `γ^{(h)} ∈ R^{2K+2}` indexed by `adj_role`. That makes `K` a weight shape
    and is *undefined* on an unseen database — unrecoverable, not merely miscalibrated. Here the role
    id only ever indexes the **frozen** name table, and the learned parts are `v^(h) ∈ R^{d_sig}`
    plus three directions.

    Memory contract: the per-`(head, role)` scalar table is built **once per forward** and *gathered*
    with `adj_role`, never materialised as a `[B,H,R,R]` parameter. `[B,H,R,R]` at `B=64,H=8,R=160`
    fp32 is 52 MB/block — ~2% of a block's attention memory, since the cell attention at `S=512` is
    10.2× larger per score matrix.
    """

    def __init__(
        self,
        d_model: int,
        d_sig: int,
        role_name_emb: Tensor,
        ladder: TimeLadder,
        *,
        n_heads: int = 8,
        role_bias: str = "name_derived",
        time_bias: str = "rope",
    ):
        super().__init__()
        if role_bias not in ("name_derived", "none"):
            raise ValueError(f"unknown role_bias {role_bias!r} (id_lookup is REMOVED, see §3.5)")
        if time_bias not in ("rope", "none", "fixed_basis"):
            raise ValueError(f"unknown time_bias {time_bias!r}")
        _frozen(self, "role_name_emb", role_name_emb)
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.role_bias = role_bias
        self.time_bias = time_bias
        self.norm = RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)

        if role_bias == "name_derived":
            self.w_rho = nn.Linear(role_name_emb.shape[-1], d_sig, bias=False)
            self.v_head = nn.Parameter(torch.randn(n_heads, d_sig) * d_sig ** -0.5)
            self.c_dir = nn.Parameter(torch.zeros(n_heads, N_DIR))
        if time_bias == "fixed_basis":
            # the Phase 1 `t3b` arm: additive bias over the FIXED ladder. Zero dataset constants, so
            # legal under §0 — the artifact was never the additive *form*, it was learned ω / fitted μ,σ.
            self.w_tb = nn.Parameter(torch.zeros(n_heads, 2 * ladder.n_freq))

    def _gamma(self, cb) -> Tensor | None:
        """``[B, H, R, R]`` role bias, gathered from a per-(head, role) table. None if disabled."""
        if self.role_bias == "none":
            return None
        K1 = self.role_name_emb.shape[0]                                   # K+1 (row 0 = FK_NONE)
        # per-(head, role) scalars: [H, K+1]. Built once per forward, then gathered.
        tab = self.v_head @ self.w_rho(self.role_name_emb).t()             # [H, K+1]

        adj = cb.adj_role                                                  # [B,R,R]
        K = (K1 - 1)
        # adj encoding: 0 none | 1..K child | K+1..2K parent | 2K+1 self
        role_ix = torch.where(adj > K, adj - K, adj).clamp(0, K1 - 1)
        role_ix = torch.where(adj == 2 * K + 1, torch.zeros_like(adj), role_ix)
        g = tab[:, role_ix.reshape(-1)].view(self.h, *adj.shape).permute(1, 0, 2, 3)

        dirs = torch.full_like(adj, DIR_CHILD)
        dirs = torch.where(adj > K, torch.full_like(adj, DIR_PARENT), dirs)
        dirs = torch.where(adj == 2 * K + 1, torch.full_like(adj, DIR_SELF), dirs)
        c = self.c_dir[:, dirs.reshape(-1)].view(self.h, *adj.shape).permute(1, 0, 2, 3)
        return g + c

    def forward(self, u: Tensor, s: Tensor, cb) -> tuple[Tensor, dict]:
        B, R, _ = u.shape
        x = self.norm(u)
        q = self.w_q(x).view(B, R, self.h, self.d_h).transpose(1, 2)       # [B,H,R,dh]
        k = self.w_k(x).view(B, R, self.h, self.d_h).transpose(1, 2)
        v = self.w_v(x).view(B, R, self.h, self.d_h).transpose(1, 2)

        tau = self.ladder.tau_from_times(cb.seed_time.unsqueeze(1), cb.row_time_r)
        theta = self.ladder.theta(tau, cb.row_is_timed)                    # [B,R,n_freq]
        if self.time_bias == "rope":
            th = theta.unsqueeze(1)                                        # broadcast over heads
            q = self.ladder.rotate(q, th, self.n_rot)
            k = self.ladder.rotate(k, th, self.n_rot)

        scores = (q @ k.transpose(-2, -1)) / self.d_h ** 0.5               # [B,H,R,R]

        gamma = self._gamma(cb)
        if gamma is not None:
            scores = scores + gamma.to(scores.dtype)
        if self.time_bias == "fixed_basis":
            d = theta.unsqueeze(2) - theta.unsqueeze(1)                    # [B,R,R,n_freq]
            basis = torch.cat([torch.sin(d), torch.cos(d)], dim=-1)
            scores = scores + torch.einsum("hf,brsf->bhrs", self.w_tb, basis.to(scores.dtype))

        # theta=0 alone would read as "Delta=0, maximally recent" for BOTH untimed rows and rows
        # whose Delta was clamped (§9.10); only additive flags can separate the three states.
        clamped = self.ladder.was_clamped(cb.seed_time.unsqueeze(1), cb.row_time_r) & cb.row_is_timed
        scores = scores + self.ladder.time_bias(cb.row_is_timed, cb.row_is_timed,
                                                clamped, clamped).unsqueeze(1).to(scores.dtype)

        valid = cb.row_valid
        mask = (cb.adj_role != 0) & valid.unsqueeze(2) & valid.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        dead = ~mask.any(-1)                                               # [B,R] — padding rows
        a = torch.softmax(scores, dim=-1)
        a = torch.where(dead.view(B, 1, R, 1), torch.zeros_like(a), a)

        out = (a @ v).transpose(1, 2).reshape(B, R, -1)
        diag = {"row_attn_entropy": _entropy(a, mask)}
        if gamma is not None:
            # |gamma| per role and per direction (§7). Near-zero magnitude means Phase 2 is a null
            # for a MECHANICAL reason, which we want to know at step 1, not after 27 runs.
            diag["gamma_abs_mean"] = gamma.detach().abs().mean()
            diag["gamma_abs_per_dir"] = self.c_dir.detach().abs().mean(0)
        return u + self.w_o(out), diag


class Broadcast(nn.Module):
    """§3.6 — high→low. ``h_i += W_b u_{ν(i)}``; FiLM variant behind a flag, off by default."""

    def __init__(self, d_model: int, *, mode: str = "additive"):
        super().__init__()
        if mode not in ("additive", "film", "none"):
            raise ValueError(f"unknown broadcast mode {mode!r}")
        self.mode = mode
        if mode == "additive":
            self.w_b = nn.Linear(d_model, d_model, bias=False)
        elif mode == "film":
            self.w_b = nn.Linear(d_model, 2 * d_model, bias=False)

    def forward(self, h: Tensor, u: Tensor, cb) -> Tensor:
        if self.mode == "none":
            return h
        idx = cb.cell_row.clamp_min(0)                                     # [B,S]
        gathered = torch.gather(u, 1, idx.unsqueeze(-1).expand(-1, -1, u.shape[-1]))
        if self.mode == "additive":
            return h + self.w_b(gathered)
        gamma, beta = self.w_b(gathered).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class RowToCellAttention(nn.Module):
    r"""§3.6 as **attention** — high→low with the cell choosing which row to read.

    .. math::
        s^{(h)}_{i s} = \frac{\tilde q_i^{(h)\top}\tilde k_s^{(h)}}{\sqrt{d_h}}
                        + \gamma^{(h)}_{\nu(i)s} + b_{untimed}\mathbb{1}[\cdot]
        \qquad h_i \mathrel{+}= W_o \sum_s \mathrm{softmax}(s)_{is}\, W_v u_s

    Cells are queries, row tokens are keys and values. Cell ``i`` sits in row ``ν(i)`` and may attend
    to row ``s`` iff ``adj_role[ν(i), s] != 0`` — which always contains the self-loop, so a cell can
    never lose sight of its own row — AND ``s`` is a valid row.

    **What this fixes.** :class:`Broadcast` adds ``W_b u_{ν(i)}`` to every cell of a row: one vector,
    identical for all of them, and it is the cell's *own* row's state. Neighbour information does
    reach it — row attention mixed the neighbours in at step 4 — but only pre-averaged, with no way
    for a cell to ask for a *particular* neighbour. A cell holding ``driver.code`` cannot weight its
    parent ``race`` row differently from a sibling ``result`` row, because it does not see them as
    separate objects at all. Here it does, and the same name-derived ``γ`` as
    :class:`RowAttention` rides on the logits so parent/child/self stay distinguishable.

    This is the symmetric counterpart of :class:`RowPool`: pool is cells→row with row-derived
    queries, this is rows→cell with cell-derived queries.

    **Cost.** Scores are ``[B,H,S,R]``, not ``[B,H,S,S]`` — ``R/S`` of a cell-attention score matrix
    (4.6% at ``S=3456, R=160``). The mask and ``γ`` both depend only on ``(ν(i), s)``, so they are
    computed once at row×row and *gathered* onto cells rather than built at cell resolution.

    The diagnostic to read first is ``r2c_own_row_mass``. If it sits at ~1.0 the cell is ignoring its
    neighbours and this operator has collapsed back to :class:`Broadcast` — a null for a mechanical
    reason, which is worth knowing before interpreting any metric.
    """

    def __init__(
        self,
        d_model: int,
        d_sig: int,
        role_name_emb: Tensor,
        ladder: TimeLadder,
        *,
        n_heads: int = 8,
        role_bias: str = "name_derived",
        time_bias: str = "rope",
    ):
        super().__init__()
        if role_bias not in ("name_derived", "none"):
            raise ValueError(f"unknown role_bias {role_bias!r}")
        if time_bias not in ("rope", "none", "fixed_basis"):
            raise ValueError(f"unknown time_bias {time_bias!r}")
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        _frozen(self, "role_name_emb", role_name_emb)
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.role_bias = role_bias
        self.time_bias = time_bias
        self.norm_h = RMSNorm(d_model)
        self.norm_u = RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)
        if role_bias == "name_derived":
            # Its own γ, not RowAttention's: the two operators ask different questions of the same
            # role ("which neighbour should this ROW mix in" vs "which row should this CELL read"),
            # and tying them would silently constrain one to the other's answer.
            self.w_rho = nn.Linear(role_name_emb.shape[-1], d_sig, bias=False)
            self.v_head = nn.Parameter(torch.randn(n_heads, d_sig) * d_sig ** -0.5)
            self.c_dir = nn.Parameter(torch.zeros(n_heads, N_DIR))

    def _gamma_rows(self, cb) -> Tensor | None:
        """``[B,H,R,R]`` role bias at ROW resolution — gathered onto cells by the caller."""
        if self.role_bias == "none":
            return None
        K1 = self.role_name_emb.shape[0]
        tab = self.v_head @ self.w_rho(self.role_name_emb).t()             # [H, K+1]
        adj = cb.adj_role                                                  # [B,R,R]
        K = K1 - 1
        role_ix = torch.where(adj > K, adj - K, adj).clamp(0, K1 - 1)
        role_ix = torch.where(adj == 2 * K + 1, torch.zeros_like(adj), role_ix)
        g = tab[:, role_ix.reshape(-1)].view(self.h, *adj.shape).permute(1, 0, 2, 3)
        dirs = torch.full_like(adj, DIR_CHILD)
        dirs = torch.where(adj > K, torch.full_like(adj, DIR_PARENT), dirs)
        dirs = torch.where(adj == 2 * K + 1, torch.full_like(adj, DIR_SELF), dirs)
        c = self.c_dir[:, dirs.reshape(-1)].view(self.h, *adj.shape).permute(1, 0, 2, 3)
        return g + c

    def forward(self, h: Tensor, u: Tensor, cb) -> tuple[Tensor, dict]:
        B, S, _ = h.shape
        R = u.shape[1]
        row_of = cb.cell_row.clamp(0, R - 1)                               # [B,S]; pad -> row 0
        cell_ok = (~cb.is_padding) & (cb.cell_row >= 0) & (cb.cell_row < R)

        hq, uk = self.norm_h(h), self.norm_u(u)
        q = self.w_q(hq).view(B, S, self.h, self.d_h).transpose(1, 2)      # [B,H,S,dh]
        k = self.w_k(uk).view(B, R, self.h, self.d_h).transpose(1, 2)      # [B,H,R,dh]
        v = self.w_v(uk).view(B, R, self.h, self.d_h).transpose(1, 2)

        # A cell's time IS its row's time (the collate contract), so the cell-side ladder input is
        # its row's theta gathered by `row_of` — the rotation is then exactly the row↔row one, and a
        # cell inherits the temporal geometry of the row it belongs to rather than inventing a second.
        tau = self.ladder.tau_from_times(cb.seed_time.unsqueeze(1), cb.row_time_r)
        theta_r = self.ladder.theta(tau, cb.row_is_timed)                  # [B,R,n_freq]
        if self.time_bias == "rope":
            theta_c = theta_r.gather(1, row_of.unsqueeze(-1).expand(-1, -1, theta_r.shape[-1]))
            q = self.ladder.rotate(q, theta_c.unsqueeze(1), self.n_rot)
            k = self.ladder.rotate(k, theta_r.unsqueeze(1), self.n_rot)

        scores = (q @ k.transpose(-2, -1)) / self.d_h ** 0.5               # [B,H,S,R]

        gamma = self._gamma_rows(cb)
        if gamma is not None:
            idx = row_of.view(B, 1, S, 1).expand(B, self.h, S, R)
            scores = scores + gamma.gather(2, idx).to(scores.dtype)

        clamped_r = (self.ladder.was_clamped(cb.seed_time.unsqueeze(1), cb.row_time_r)
                     & cb.row_is_timed)
        timed_c = cb.row_is_timed.gather(1, row_of)                        # [B,S]
        clamped_c = clamped_r.gather(1, row_of)
        scores = scores + self.ladder.time_bias(timed_c, cb.row_is_timed,
                                                clamped_c, clamped_r).unsqueeze(1).to(scores.dtype)

        row_mask = (cb.adj_role != 0) & cb.row_valid.unsqueeze(1)          # [B,R,R] (q-row, k-row)
        mask = row_mask.gather(1, row_of.unsqueeze(-1).expand(-1, -1, R))  # [B,S,R]
        mask = mask & cell_ok.unsqueeze(-1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        dead = ~mask.any(-1)                                               # [B,S] pad / orphan cells
        a = torch.softmax(scores, dim=-1)
        a = torch.where(dead.view(B, 1, S, 1), torch.zeros_like(a), a)

        out = (a @ v).transpose(1, 2).reshape(B, S, -1)
        own = F.one_hot(row_of, R).to(a.dtype).unsqueeze(1)                # [B,1,S,R]
        live = (~dead).view(B, 1, S, 1)
        # detached: these are read, never optimised, and keeping them in the graph would pin `a`
        ad = a.detach()
        diag = {"r2c_entropy": _entropy(ad, mask),
                # the number that says whether this operator is doing anything Broadcast could not
                "r2c_own_row_mass": ((ad * own).sum(-1, keepdim=True) * live).sum()
                                    / live.expand_as(ad[..., :1]).sum().clamp_min(1)}
        if gamma is not None:
            diag["r2c_gamma_abs_mean"] = gamma.detach().abs().mean()
        return h + self.w_o(out), diag


class RowMoE(nn.Module):
    r"""§3.7 — row-level MoE, routed on the value-free row signature.

    .. math::
        \mathrm{logits}_e = \frac{1}{T}\frac{\langle W_g^{(e)}, z_r\rangle}{\|W_g^{(e)}\|\|z_r\|},
        \qquad u_r \mathrel{+}= \sum_{e\in\mathrm{top}k} g_e E_e(\mathrm{RMSNorm}(u_r))

    Cosine routing with a **learned** temperature `T` (init 1.0): the orthogonality penalty
    constrains expert *directions* but leaves logit *scale* free, so scale needs its own parameter.

    This is **added alongside** the existing cell-level ``MoEFFN``, not substituted for it — one
    mechanism at two granularities, each routing on the value-free signature of its own object.
    ``moe.py`` is deliberately left untouched so the cell MoE keeps behaving byte-identically; only
    :class:`~gloss.model.moe.SwiGLU` is reused here.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_sig: int,
        *,
        num_experts: int = 4,
        k: int = 2,
        use_shared: bool = True,
        lambda_ortho: float = 0.5,
        lambda_balance: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.lambda_ortho = lambda_ortho
        self.lambda_balance = lambda_balance
        self.norm = RMSNorm(d_model)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        # SHARED + ROUTED (DeepSeek-MoE style), and ON by default at the row level. The shared expert
        # is always applied, so the routed experts only have to model what is *specific* to a row's
        # signature rather than re-learning the shared transform in every expert. Note the cell level
        # keeps `use_shared=False` (its `+S` arm measured as only mildly positive, on regression
        # alone), so the two levels deliberately differ here.
        self.shared = SwiGLU(d_model, d_ff) if use_shared else None
        self.w_g = nn.Parameter(torch.randn(num_experts, d_sig) * d_sig ** -0.5)
        self.log_T = nn.Parameter(torch.zeros(()))                         # T = exp(0) = 1.0

    def gates(self, z: Tensor) -> Tensor:
        logits = (F.normalize(z, dim=-1) @ F.normalize(self.w_g, dim=-1).t()) / self.log_T.exp()
        topv, topi = logits.topk(self.k, dim=-1)
        masked = torch.full_like(logits, float("-inf")).scatter_(-1, topi, topv)
        return torch.softmax(masked, dim=-1)

    def ortho_loss(self) -> Tensor:
        W = F.normalize(self.w_g, dim=-1)
        gram = W @ W.t()
        eye = torch.eye(self.num_experts, device=W.device, dtype=W.dtype)
        return ((gram - eye) ** 2).sum()

    def balance_loss(self, g: Tensor, valid: Tensor) -> Tensor:
        """``M·Σ_e f_e p_e`` over VALID rows only — padding rows must not vote."""
        m = valid.reshape(-1)
        gv = g.reshape(-1, self.num_experts)[m]
        if gv.numel() == 0:
            return g.sum() * 0.0
        p = gv.mean(0)
        top1 = gv.argmax(-1)
        f = torch.bincount(top1, minlength=self.num_experts).to(gv.dtype) / gv.shape[0]
        return self.num_experts * (f * p).sum()

    def forward(self, u: Tensor, z: Tensor, cb) -> tuple[Tensor, Tensor, dict]:
        g = self.gates(z)
        x = self.norm(u)
        y = torch.zeros_like(u)
        for e, expert in enumerate(self.experts):                          # dense combine (MVP)
            y = y + g[..., e:e + 1] * expert(x)
        if self.shared is not None:
            y = y + self.shared(x)                                         # always-on, ungated

        valid = cb.row_valid
        aux = self.lambda_ortho * self.ortho_loss() \
            + self.lambda_balance * self.balance_loss(g, valid)

        gv = g.reshape(-1, self.num_experts)[valid.reshape(-1)]
        diag = {
            "row_expert_usage": (torch.bincount(gv.argmax(-1), minlength=self.num_experts)
                                 .to(g.dtype) / max(gv.shape[0], 1)).detach()
            if gv.numel() else torch.zeros(self.num_experts, device=g.device),
            "row_router_norms": self.w_g.detach().norm(dim=-1),
            "row_ortho": self.ortho_loss().detach(),
            "row_balance": self.balance_loss(g, valid).detach(),
            "row_T": self.log_T.detach().exp(),
        }
        return u + y, aux, diag


def _entropy(a: Tensor, mask: Tensor) -> Tensor:
    """Mean attention entropy over rows that have at least one admissible neighbour."""
    live = mask.any(-1)                                                    # [B,R]
    p = a.clamp_min(1e-9)
    ent = -(p * p.log()).sum(-1)                                           # [B,H,R]
    keep = live.unsqueeze(1).expand_as(ent)
    return (ent * keep).sum() / keep.sum().clamp_min(1) if keep.any() else ent.sum() * 0.0
