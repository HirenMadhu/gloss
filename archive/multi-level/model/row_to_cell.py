"""Retired: `RowToCellAttention` (changes.md §3.6 as attention).

Measured against the `additive` Broadcast it replaced (results/r2c_attn, 3 seeds x 8 tasks):
worse on 7 of 8, and user-ignore collapsed to 76.34 +/- 6.23 against a 83.44 baseline. The
operator is kept here because `results/r2c_attn/` cannot be regenerated without it.

Imports below are the ones the class used from gloss.model.row_level.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gloss.model.row_level import RMSNorm, _entropy, _frozen  # noqa: F401


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


