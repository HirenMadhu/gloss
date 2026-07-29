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
        self.w_tau = nn.Linear(2 * ladder.n_freq, d_sig, bias=False)
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

        tau = self.ladder.tau_from_times(cb.seed_time.unsqueeze(1), cb.row_time_r)
        z = z + self.w_tau(self.ladder.feats(tau, cb.row_is_timed).to(z.dtype))
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
        """``[B, R, S]`` bool — cell `i` belongs to row `r` and is not padding."""
        rows = torch.arange(R, device=cb.cell_row.device).view(1, R, 1)
        return (cb.cell_row.unsqueeze(1) == rows) & (~cb.is_padding).unsqueeze(1)

    def forward(self, h: Tensor, u: Tensor, s: Tensor, cb) -> Tensor:
        """``h [B,S,d]``, ``u [B,R,d]``, ``s [B,R,d_sig]`` -> updated ``u [B,R,d]`` (residual)."""
        B, R, _ = u.shape
        member = self._membership(cb, R)                                  # [B,R,S]

        if self.mode == "mean":
            cnt = member.sum(-1, keepdim=True).clamp_min(1)
            return u + (member.to(h.dtype) @ h) / cnt

        q_in = {"signature": s, "hidden": u, "hybrid": torch.cat([u, s], dim=-1)}[self.mode]
        q = self.w_q(q_in).view(B, R, self.slots, self.d_h)               # [B,R,M,dh]
        k = self.w_k(self.col_name_emb[cb.col_idxs.clamp_min(0)])         # [B,S,dh]
        v = self.w_v(h)                                                   # [B,S,dh]

        scores = torch.einsum("brmd,bsd->brms", q, k) / self.d_h ** 0.5
        scores = scores.masked_fill(~member.unsqueeze(2), float("-inf"))
        # a row with no cells would be all -inf -> NaN after softmax; zero it explicitly
        empty = ~member.any(-1)                                           # [B,R]
        a = torch.softmax(scores, dim=-1)
        a = torch.where(empty.view(B, R, 1, 1), torch.zeros_like(a), a)

        pooled = torch.einsum("brms,bsd->brmd", a, v).reshape(B, R, self.slots * self.d_h)
        return u + self.w_o(pooled)


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

        # `b_untimed`: theta=0 alone would read as "Delta=0, maximally recent", which is wrong
        scores = scores + self.ladder.untimed_bias(cb.row_is_timed, cb.row_is_timed) \
            .unsqueeze(1).to(scores.dtype)

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
