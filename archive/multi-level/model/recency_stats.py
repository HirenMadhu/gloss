r"""The recency order-statistic channel (``x``) — a row-level read of the truncation window.

The sampler takes ``num_neighbors=[12, 12]`` with ``temporal_strategy="last"``, so a role whose true
fanout exceeds 12 delivers the 12 **most recent** children and no signal that more existed;
``num_rows`` saturates at the cap. Under "last", the *span* of what survived encodes an arrival rate:
a role whose 12 children span an hour is a faster relation than one whose 12 span a decade. Nothing in
the substrate computes an order statistic over recency, so that signal sits in the batch unread. This
module is the operator that reads it.

Per query row ``r`` and role ``rho``, over the sampled child set ``C(r, rho)``:

.. math::
    \ell_c = \log(1 + \Delta_c),\quad \Delta_c = t^* - t_c \ \text{(seconds)}

    x_{\max} = \sum_c \mathrm{softmax}_c(\kappa_{\max}\ell)_c\,\ell_c, \qquad
    x_{\min} = \sum_c \mathrm{softmax}_c(\kappa_{\min}\ell)_c\,\ell_c

with two learnable scalars ``kappa_max`` (init +4) and ``kappa_min`` (init −4), global — not per-role
and not per-database. Large ``|kappa|`` approaches a hard max/min; ``kappa -> 0`` collapses to the
mean, which is the failure mode to watch: the probe evidence came from a *hard* max, and a gate that
settles on the bulk smooths away the tail that carries the signal. ``kappa`` is logged for that reason.

Three free flags per role complete the feature: ``sat`` (``|C| >= w`` — the max age only means *rate*
when the window is truncated; an unsaturated role's max age is the entity's full history, a different
quantity), ``exists``, and ``untimed``. Aggregation over roles is a permutation-invariant sum of an MLP
over ``[x_max, x_min, sat, exists, untimed] (+) e_rho``, so ``K`` can differ across databases, and
``e_rho`` is a projection of the **frozen role-name embedding** — never a ``K``-shaped parameter (§0).

**Hard constraints, all structural rather than by convention:**

* ``h_x`` enters the **value path only** — it is added to the row token ``u``. It never touches either
  router: the row router reads ``s`` (the row signature: table (+) in-role (+) hop (+) recency), which
  this module does not construct, read, or modify. Cardinality is a *neighbourhood* statistic and would
  break the leak-free routing property that ``test_routing_invariance.py`` pins.
* Zero per-database parameters: ``kappa_max``, ``kappa_min``, the MLP and ``alpha`` are global scalars
  or fixed-width matrices.
* ``alpha`` inits at **0**, so an ``x_full`` run starts numerically identical to ``base``.
* Untimed child tables take the null path (``x = 0``, ``untimed = 1``). Delta is never imputed.

**On the causality assert.** The spec asks for a loud failure if any ``Delta_c < 0``. Measured with
``scripts/probe_role_window.py`` before this module existed: **0 violations in 7,474 child rows** over
rel-f1 / rel-trial / rel-event — but rel-event carries a root row with ``Delta < 0``, which is the
already-documented §9.10 clamped-row class (``TimeLadder.was_clamped`` / ``b_clamped`` exist precisely
because those rows are real). So the assert is scoped to **child rows**, the only rows this channel
aggregates over. Scoped to all rows it would fire on a known-and-handled condition and take every
rel-event run down with it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .row_level import RMSNorm, _frozen
from .time_encoding import TimeLadder

#: Sampler fanout cap in effect (``gloss/train/datamodule.py``: ``num_neighbors or [12, 12]``).
#: Saturation is ``>=``, not ``==``: rel-event role 1 reaches 13 children because a row can be
#: reached as a child via two different parents, so the union exceeds the per-hop cap.
DEFAULT_W = 12

MODES = ("full", "flags", "shuffle")


class RecencyOrderChannel(nn.Module):
    """``forward(cb) -> (h_x [B, R, d_model], diag)``. Add ``h_x`` to the row token; it is already
    scaled by ``alpha`` (init 0), so a fresh model is bit-identical to one without the channel.

    ``mode`` selects the arm:

    * ``full``    — the mechanism as specified.
    * ``flags``   — ``v = [0, 0, sat, exists, untimed]``. The important control: if it matches
      ``full``, the win is "the model learned it was truncated" and the rate story is wrong.
    * ``shuffle`` — real ``v``, but the recency vector is permuted across seeds within the batch, so
      parameters and magnitudes are identical and only the row-to-Delta binding is destroyed. Catches
      "any extra feature with these parameters helps".
    """

    def __init__(
        self,
        d_model: int,
        role_name_emb: Tensor,
        *,
        d_role: int = 32,
        d_hidden: int = 64,
        w: int = DEFAULT_W,
        mode: str = "full",
        kappa_max_init: float = 4.0,
        kappa_min_init: float = -4.0,
        assert_causal: bool = True,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown recency-channel mode {mode!r}; expected one of {MODES}")
        _frozen(self, "role_name_emb", role_name_emb)          # [K+1, d_text], row 0 = FK_NONE
        self.num_roles = int(role_name_emb.shape[0]) - 1
        self.mode = mode
        self.w = int(w)
        self.assert_causal = assert_causal
        self.w_role = nn.Linear(role_name_emb.shape[-1], d_role, bias=False)
        self.mlp = nn.Sequential(nn.Linear(5 + d_role, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_model))
        self.norm = RMSNorm(d_model)
        # global scalars: no per-role, no per-database parameter anywhere in this module
        self.kappa_max = nn.Parameter(torch.tensor(float(kappa_max_init)))
        self.kappa_min = nn.Parameter(torch.tensor(float(kappa_min_init)))
        self.alpha = nn.Parameter(torch.zeros(()))

    # -- soft order statistic over ragged groups -------------------------------------------------
    @staticmethod
    def _soft_stat(kappa: Tensor, ell: Tensor, member: Tensor, gid: Tensor, n_groups: int) -> Tensor:
        """``sum_c softmax_c(kappa * ell_c) * ell_c`` per group -> ``[n_groups]``.

        ``ell``/``member``/``gid`` are all ``[B, R, R]`` over (query row, candidate child); ``gid``
        indexes the flattened ``(b, r, rho)`` group. Non-members are excluded from the softmax rather
        than given a large negative score, so an empty group yields 0 instead of NaN.
        """
        score = torch.where(member, kappa * ell, torch.full_like(ell, float("-inf")))
        flat_s, flat_g = score.reshape(-1), gid.reshape(-1)
        # shift by the per-group max for stability; detached, since softmax is invariant to it
        gmax = torch.full((n_groups,), float("-inf"), device=ell.device, dtype=ell.dtype)
        gmax = gmax.scatter_reduce(0, flat_g, flat_s.detach(), reduce="amax", include_self=True)
        gmax = torch.where(torch.isfinite(gmax), gmax, torch.zeros_like(gmax))
        e = torch.where(member.reshape(-1), torch.exp(flat_s - gmax[flat_g]),
                        torch.zeros_like(flat_s))
        den = torch.zeros(n_groups, device=ell.device, dtype=ell.dtype).scatter_add(0, flat_g, e)
        num = torch.zeros(n_groups, device=ell.device, dtype=ell.dtype).scatter_add(
            0, flat_g, e * ell.reshape(-1))
        return num / den.clamp_min(1e-9)

    def forward(self, cb) -> tuple[Tensor, dict]:
        dev = cb.row_valid.device
        B, R = cb.row_valid.shape
        K = self.num_roles
        G = K + 1                                    # group slot 0 is the discard bin

        # ---- recency of every row, in the substrate's own unit (seconds, log1p) ----
        seed_t = cb.seed_time.unsqueeze(1)
        raw = seed_t.to(torch.float64) - cb.row_time_r.to(torch.float64)          # [B,R], pre-clamp
        ell = TimeLadder.tau(raw).to(torch.float32)                               # [B,R]

        # ---- child sets: s is a CHILD of r via role s->r iff adj_role[b,r,s] in [1,K] ----
        adj = cb.adj_role
        valid_pair = cb.row_valid.unsqueeze(2) & cb.row_valid.unsqueeze(1)        # [B,R,R]
        is_child = (adj >= 1) & (adj <= K) & valid_pair
        timed_s = cb.row_is_timed.unsqueeze(1).expand(B, R, R)                    # child's timedness

        if self.assert_causal:
            # child rows only — root rows with Delta < 0 are the known §9.10 class, not a hole here
            raw_s = raw.unsqueeze(1).expand(B, R, R)          # child's Delta, per (query row, child)
            bad = is_child & timed_s & (raw_s < 0)
            if bool(bad.any()):
                n, worst = int(bad.sum()), float(raw_s[bad].min())
                raise AssertionError(
                    f"temporal constraint violated: {n} sampled child rows have row_time > seed_time "
                    f"(worst Delta = {worst:.1f}s). The sampler's causal guarantee has a hole and "
                    f"nothing downstream of it is trustworthy. Measured 0/7474 when this channel was "
                    f"written (scripts/probe_role_window.py); re-run that probe before weakening this."
                )

        role = torch.where(is_child, adj, torch.zeros_like(adj))                  # [B,R,R] in [0,K]
        base = (torch.arange(B * R, device=dev) * G).view(B, R, 1)
        gid = (base + role).reshape(B, R, R)                                      # flat (b,r,rho)
        n_groups = B * R * G

        ones = torch.ones(B * R * R, device=dev, dtype=torch.float32)
        flat_g = gid.reshape(-1)
        count = torch.zeros(n_groups, device=dev).scatter_add(
            0, flat_g, torch.where(is_child.reshape(-1), ones, torch.zeros_like(ones)))
        n_timed = torch.zeros(n_groups, device=dev).scatter_add(
            0, flat_g, torch.where((is_child & timed_s).reshape(-1), ones, torch.zeros_like(ones)))

        exists = (count > 0).to(torch.float32)
        sat = (count >= self.w).to(torch.float32) * exists
        untimed = ((n_timed == 0) & (count > 0)).to(torch.float32)

        # ---- the two soft order statistics ----
        if self.mode == "flags":
            x_max = torch.zeros(n_groups, device=dev)
            x_min = torch.zeros(n_groups, device=dev)
        else:
            ell_src = ell
            if self.mode == "shuffle":
                # placebo: same parameters and in-distribution magnitudes, binding to the row broken.
                # Deterministic under the global seed via torch's own generator.
                ell_src = ell[torch.randperm(B, device=dev)]
            ell_bc = ell_src.unsqueeze(1).expand(B, R, R)
            member = is_child & timed_s
            x_max = self._soft_stat(self.kappa_max, ell_bc, member, gid, n_groups)
            x_min = self._soft_stat(self.kappa_min, ell_bc, member, gid, n_groups)

        live = (1.0 - untimed) * exists                     # null path: untimed or empty -> x = 0
        x_max, x_min = x_max * live, x_min * live

        # ---- aggregate over roles, skipping absent (r, rho) pairs ----
        # An absent role would otherwise contribute MLP(0 (+) e_rho), a constant per role summed over
        # every row -- work proportional to B*R*K for no signal. Only non-empty groups are gathered.
        v = torch.stack([x_max, x_min, sat, exists, untimed], dim=-1)             # [n_groups, 5]
        keep = (exists > 0).nonzero(as_tuple=False).squeeze(-1)
        h_x = torch.zeros(B * R, cb_dim(self.mlp), device=dev, dtype=v.dtype)
        if keep.numel():
            rho = keep % G
            row_flat = keep // G
            e_rho = self.w_role(self.role_name_emb[rho.clamp(0, K)])
            h_x = h_x.index_add(0, row_flat, self.mlp(torch.cat([v[keep], e_rho], dim=-1)))
        h_x = h_x.view(B, R, -1) * cb.row_valid.unsqueeze(-1).to(h_x.dtype)

        # Per-role breakdown (§5): the offline census predicts which roles have a window to read;
        # this is what actually happened at runtime, which is the thing to check the prediction
        # against. Reduced over (b, r) so the lists are K+1 long and role-indexed.
        gv = lambda t: t.view(B * R, G).sum(0)                    # noqa: E731
        n_role = gv(exists)
        per_role = {
            "x_role_groups": n_role.tolist(),
            "x_role_sat_rate": (gv(sat) / n_role.clamp_min(1)).tolist(),
            "x_role_untimed_rate": (gv(untimed) / n_role.clamp_min(1)).tolist(),
            "x_role_max_mean": (gv(x_max.detach() * live) / gv(live).clamp_min(1)).tolist(),
        }

        diag = {
            **per_role,
            "x_kappa_max": float(self.kappa_max.detach()),
            "x_kappa_min": float(self.kappa_min.detach()),
            "x_alpha": float(self.alpha.detach()),
            "x_sat_rate": float((sat.sum() / exists.sum().clamp_min(1)).detach()),
            "x_untimed_rate": float((untimed.sum() / exists.sum().clamp_min(1)).detach()),
            "x_groups_per_row": float((exists.sum() / max(int(cb.row_valid.sum()), 1)).detach()),
            "x_max_mean": float((x_max.sum() / live.sum().clamp_min(1)).detach()),
            "x_max_std": float(x_max[live > 0].std().detach()) if float(live.sum()) > 1 else 0.0,
        }
        return self.alpha * self.norm(h_x), diag


def cb_dim(mlp: nn.Sequential) -> int:
    """Output width of the aggregation MLP (its last ``Linear``)."""
    return mlp[-1].out_features
