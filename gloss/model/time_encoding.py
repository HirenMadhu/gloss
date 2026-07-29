"""The fixed-frequency temporal ladder — the *only* place time enters the model (changes.md §3.1).

Canonical unit is **seconds, always, on every database**:

    Δ_r = max(0, t* − t_r)   [seconds],      τ_r = log(1 + Δ_r)

Seconds is a physical unit, not a fitted one: every database that has ever existed lands in
``τ ∈ [0, 22]`` (one second is 0.69, one century is 21.9), so the range is known before a single row is
read and there is **nothing to calibrate**. There is no standardisation, no ``μ_τ``, no ``σ_τ``, no
per-database unit and no calibration file — those are exactly the illegal artifacts of changes.md §0's
no-dataset-artifact rule (*"a tensor may depend on the data only if it is recomputable from a new
database without gradients"*). Two facts make the calibration unnecessary rather than merely
inconvenient:

* ``μ_τ`` **cancels exactly.** Rotary scores depend on ``τ_i − τ_j``; an additive shift in τ is invisible
  to the model (unit-tested: ``test_relative_only``).
* ``σ_τ`` is replaced by a **wide fixed band.** The ladder spans the whole ``[0, 22]`` range at multiple
  resolutions and each database reads whichever channels carry signal for its own spread — the standard
  multi-resolution RoPE argument, and the reason RoPE generalises without per-corpus statistics.

**The ladder.** ``n_freq = 8`` frequencies ``ω_k``, log-spaced over ``[ω_min, ω_max] = [0.3, 5.0]``,
held in a **non-persistent buffer, never an** ``nn.Parameter`` **and never learned**:

    θ_r^(k) = ω_k · τ_r

**Band note — this was MEASURED, and it corrects changes.md.** The original ``[0.05, 5.0]`` was derived
from τ's *range* ``[0, 22]``. What the frequencies must resolve is τ's **spread**, which is an order of
magnitude smaller: measured over 768 seeds/dataset, τ has mean 14–18 but **σ ≈ 1.59–1.90**. At
``ω = 0.05`` that is ``ω·σ = 0.08`` rad of variation across the entire corpus — so the two lowest
channels were dead weight (``var(sin ωτ) < 0.01`` on rel-f1 and rel-trial). ``ω_min`` is therefore set
from σ:

* **alive**: want ``ω·σ ≳ 0.5`` rad (the first healthy measured channel sat at 0.58) ⇒ ``ω_min ≳ 0.31``
* **no wraparound** over the bulk ⇒ ``ω_min·span < π``

Both hold at ``ω_min = 0.3``. The τ distribution is **truncated on the right** — τ cannot exceed the
database's own age — so the bulk is *not* a symmetric ±3σ interval: on rel-f1 it runs from
``mean − 3σ = 11.4`` to ``max = 19.66``, a span of 8.2, giving ``ω·span = 2.47 rad < π``. (An earlier
draft of this note assumed a symmetric ``6σ = 11.4`` span and wrongly concluded the band was
over-constrained.)

**One coupling to know about, because it is not obvious.** The *measured* wraparound check still fires
on rel-f1 at ``ω_min = 0.3``, because the full observed range is 19.7, not 8.2 — τ reaches **0**. That
τ = 0 mass is not the bulk: it is the seed/root rows whose ``Δ = max(0, t* − t_r)`` **clamps to zero**
because they are dated after their own seed time (35% of rel-event task rows; changes.md §9.10). So the
ladder band and the seed-row clamp are coupled: flag seed rows instead of clamping them and the range
collapses to the bulk, where this band has no wraparound at all. Until §9.10 is decided, the lowest
channel aliases τ ≈ 0 against τ ≈ 21 — i.e. "this is the query row" against "this is ancient".

``ω_max`` stays at 5.0: period ``≈ 1.26`` τ units, resolving recency *ratios* of about
``e^{1.26} ≈ 3.5×``, where measured ``var(sin) = 0.494`` is already saturated (0.5 is the maximum), so
going higher only aliases.

The frequencies remain **fixed constants chosen once from corpus-wide statistics**, not per-database
fitted parameters, so §0 still holds — but say so explicitly in the write-up, because "chosen by
looking at the data" invites exactly that objection. §7's per-frequency utilisation logging is the
guard that catches a transfer database whose spread sits outside this band.

**Two readouts of the same ladder.**

* :meth:`TimeLadder.rotate` — RoPE-style rotation of the leading ``2m`` dims of each head; used at the
  cell level (§3.2) and the row level (§3.5). Because the score then depends on
  ``τ_i − τ_j = log((1+Δ_i)/(1+Δ_j))``, a log time *ratio*, rescaling a database's entire time axis by a
  constant ``c`` leaves every score unchanged for ``Δ ≫ 1s``. That is scale-equivariance by construction,
  with no fitted constant doing the work; it is exact in that regime and approximate as ``Δ → 0``, where
  the ``+1`` floor bites (``test_time_unit_invariance_*``).
* :meth:`TimeLadder.feats` — the fixed pair ``[sin θ ; cos θ] ∈ R^{2·n_freq}``, for use behind a
  **learned** linear map (the row/cell signature, §3.3). The learned map is legal under §0; only the
  frequencies were ever the risk, and they are constants.

**Untimed rows** get ``θ = 0`` on all channels *plus* the learned scalar :attr:`b_untimed` added to the
attention logit whenever **either** endpoint is untimed (:meth:`untimed_bias`). One universal parameter,
no dataset dependence. ``θ = 0`` alone would collapse untimed rows onto "Δ = 0, maximally recent", which
is wrong — the explicit flag is what keeps them distinguishable (``test_untimed_is_not_delta_zero``).

**Precision.** UNIX timestamps are ~1.7e9 seconds; in float32 the spacing there is ~128 s, and after a
unit rescale it is far worse. So Δ is formed and log1p'd in **float64** (:meth:`delta_seconds`,
:meth:`tau`), matching ``CellBatch.row_time`` / ``seed_time``, which are already float64. Cast to the
model dtype only at the rotation (:meth:`rotate` casts ``cos``/``sin`` to ``x.dtype``).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# τ = log1p(Δ_seconds) for any physically plausible Δ: 1 s -> 0.69, 1 century -> 21.9.
TAU_MAX_PLAUSIBLE = 22.0


class TimeLadder(nn.Module):
    """Fixed log-spaced frequency ladder over ``τ = log1p(Δ_seconds)``, with two readouts.

    Args:
        n_freq: number of frequencies ``ω_k`` (changes.md §4 default 8).
        omega: ``(ω_min, ω_max)``, log-spaced inclusive band (default ``(0.3, 5.0)`` — MEASURED, see
            the band note below; changes.md's original ``(0.05, 5.0)`` left 2 of 8 channels dead).

    Attributes:
        omega: ``[n_freq]`` **non-persistent buffer** of fixed constants — not a parameter, absent from
            ``state_dict()``, byte-identical across any two models regardless of the data they saw.
        b_untimed: the single learned scalar added to attention logits touching an untimed endpoint.
    """

    def __init__(self, n_freq: int = 8, omega: tuple[float, float] = (0.3, 5.0)):
        super().__init__()
        omega_min, omega_max = float(omega[0]), float(omega[1])
        if not (0.0 < omega_min <= omega_max):
            raise ValueError(f"need 0 < omega_min <= omega_max, got {omega}")
        if n_freq < 1:
            raise ValueError(f"n_freq must be >= 1, got {n_freq}")
        self.n_freq = int(n_freq)
        w = (torch.logspace(math.log10(omega_min), math.log10(omega_max), self.n_freq,
                            dtype=torch.float64)
             if self.n_freq > 1 else torch.tensor([omega_min], dtype=torch.float64))
        # Non-persistent: a fixed constant of the architecture, never checkpointed, never learned.
        self.register_buffer("omega", w, persistent=False)
        self.b_untimed = nn.Parameter(torch.zeros(()))
        # changes.md §9.10. `Delta = max(0, t* - t_row)` CLAMPS every row dated after its own seed to
        # Delta = 0, i.e. tau = 0 — indistinguishable from "happened right now". Those rows are the
        # query entity's own row, which RelBench includes regardless of its timestamp: 35% of
        # rel-event/user-attendance task rows, median +9.7 days. "This is the query row" and "this is
        # maximally recent" are different statements and the clamp collapsed them.
        #
        # This is also why the ladder band trips its wraparound check: the tau = 0 mass stretches
        # rel-f1's observed range from the bulk's 8.2 to 19.7, so the lowest channel aliases tau ~ 0
        # against tau ~ 21. Flagging the clamp instead of hiding it fixes both at once.
        self.b_clamped = nn.Parameter(torch.zeros(()))

    @property
    def feat_dim(self) -> int:
        """Width of :meth:`feats` — ``2*n_freq`` plus the clamped-indicator channel."""
        return 2 * self.n_freq + 1

    # ---------------------------------------------------------------- τ ---
    @staticmethod
    def delta_seconds(seed_time: Tensor, row_time: Tensor) -> Tensor:
        """``Δ = max(0, t* − t_row)`` in seconds, computed in float64 (see the precision note)."""
        return (seed_time.to(torch.float64) - row_time.to(torch.float64)).clamp(min=0.0)

    @staticmethod
    def tau(delta_seconds: Tensor | float) -> Tensor:
        """``τ = log1p(max(0, Δ))``. Float64 throughout; finite at Δ = 0 and at any large Δ."""
        d = delta_seconds if isinstance(delta_seconds, Tensor) else torch.tensor(delta_seconds)
        return torch.log1p(d.to(torch.float64).clamp(min=0.0))

    def tau_from_times(self, seed_time: Tensor, row_time: Tensor) -> Tensor:
        """Convenience: ``τ`` straight from a seed time and a row time (broadcast as given)."""
        return self.tau(self.delta_seconds(seed_time, row_time))

    @staticmethod
    def was_clamped(seed_time: Tensor, row_time: Tensor) -> Tensor:
        """``t_row > t*`` — the rows whose Δ the ``max(0, ·)`` clamp destroyed (§9.10).

        Must be computed from the raw times, *before* clamping, which is why it cannot be recovered
        from τ downstream: after the clamp these are numerically identical to a genuine Δ = 0.
        """
        return row_time.to(torch.float64) > seed_time.to(torch.float64)

    # ------------------------------------------------------------ ladder ---
    def theta(self, tau: Tensor, is_timed: Tensor | None = None) -> Tensor:
        """``θ^(k) = ω_k · τ`` -> ``[..., n_freq]``; exactly 0 on every channel where ``is_timed`` is False."""
        th = tau.to(torch.float64).unsqueeze(-1) * self.omega.to(tau.device)
        if is_timed is not None:
            th = th * is_timed.unsqueeze(-1).to(th.dtype)
        return th

    def theta_from_times(self, seed_time: Tensor, row_time: Tensor,
                         is_timed: Tensor | None = None) -> Tensor:
        """Convenience: ``θ`` straight from times, with untimed entries zeroed."""
        return self.theta(self.tau_from_times(seed_time, row_time), is_timed)

    def feats(self, tau: Tensor, is_timed: Tensor | None = None,
              is_clamped: Tensor | None = None) -> Tensor:
        """``[sin θ ; cos θ ; 1[clamped]]`` -> ``[..., feat_dim]`` = ``[..., 2*n_freq + 1]``.

        Three states must stay linearly separable behind the learned ``W_τ`` of §3.3, and none of
        them is separable from the others by the sinusoids alone:

        * **timed, Δ > 0** — the ordinary case.
        * **untimed** — the whole pair is **zeroed**, rather than left at ``cos 0 = 1``, so it does
          not collide with Δ = 0.
        * **clamped** (``t_row > t*``, §9.10) — sinusoids are identical to a genuine Δ = 0, so the
          only thing that can distinguish it is the explicit indicator channel.
        """
        th = self.theta(tau)                                    # unmasked: mask the pair, not θ
        f = torch.cat((torch.sin(th), torch.cos(th)), dim=-1)   # [..., 2*n_freq]
        if is_timed is not None:
            f = f * is_timed.unsqueeze(-1).to(f.dtype)
        flag = (torch.zeros_like(f[..., :1]) if is_clamped is None
                else is_clamped.unsqueeze(-1).to(f.dtype))
        return torch.cat((f, flag), dim=-1)                     # [..., 2*n_freq + 1]

    # ---------------------------------------------------------- rotation ---
    def rotate(self, x: Tensor, theta: Tensor, n_rot_dims: int | None = None) -> Tensor:
        """RoPE-rotate the leading ``n_rot_dims`` of ``x``; the remaining dims pass through untouched.

        Pairs are **interleaved**: ``(x_0, x_1)`` rotates by ``θ^(0)``, ``(x_2, x_3)`` by ``θ^(1)``, …

        Args:
            x: ``[..., d]`` queries or keys (e.g. ``[B, H, S, d_h]``).
            theta: ``[..., n_freq]`` from :meth:`theta`; ``theta.shape[:-1]`` must broadcast against
                ``x.shape[:-1]`` (for per-head tensors pass e.g. ``theta[:, None]``).
            n_rot_dims: how many leading dims to rotate (even, ``<= d``, ``<= 2*theta.shape[-1]``).
                Defaults to ``2 * theta.shape[-1]`` — the whole ladder. A smaller value uses the
                **lowest** frequencies, which are the ones that stay monotonic over the τ band.

        Returns:
            ``x`` with its first ``n_rot_dims`` coordinates rotated, same shape and dtype.
        """
        n_avail = theta.shape[-1]
        n_rot = 2 * n_avail if n_rot_dims is None else int(n_rot_dims)
        if n_rot % 2 != 0:
            raise ValueError(f"n_rot_dims must be even, got {n_rot}")
        if n_rot > x.shape[-1]:
            raise ValueError(f"n_rot_dims={n_rot} exceeds x's last dim {x.shape[-1]}")
        if n_rot > 2 * n_avail:
            raise ValueError(f"n_rot_dims={n_rot} needs {n_rot // 2} frequencies, ladder has {n_avail}")
        m = n_rot // 2
        if m == 0:
            return x
        th = theta[..., :m]
        cos, sin = torch.cos(th).to(x.dtype), torch.sin(th).to(x.dtype)
        rot, rest = x[..., :n_rot], x[..., n_rot:]
        x0, x1 = rot[..., 0::2], rot[..., 1::2]
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
        return out if rest.shape[-1] == 0 else torch.cat((out, rest.expand(*out.shape[:-1],
                                                                          rest.shape[-1])), dim=-1)

    # ------------------------------------------------------------ untimed ---
    def untimed_bias(self, is_timed_q: Tensor, is_timed_k: Tensor) -> Tensor:
        """``b_untimed · 1[q or k untimed]`` -> ``[..., Sq, Sk]``, the additive logit term of §3.1/§3.5."""
        either_untimed = ~(is_timed_q.unsqueeze(-1) & is_timed_k.unsqueeze(-2))
        return self.b_untimed * either_untimed.to(self.b_untimed.dtype)

    def clamped_bias(self, is_clamped_q: Tensor, is_clamped_k: Tensor) -> Tensor:
        """``b_clamped · 1[q or k clamped]`` -> ``[..., Sq, Sk]`` (§9.10).

        The rotation cannot express this: a clamped row has ``θ = 0`` exactly like a genuine Δ = 0,
        so the *relative* angle to every other row is the same in both cases. Only an additive term
        can separate "the query row" from "maximally recent", which is the same argument §3.1 makes
        for ``b_untimed`` — hence the same mechanism, and one extra universal scalar.
        """
        either = is_clamped_q.unsqueeze(-1) | is_clamped_k.unsqueeze(-2)
        return self.b_clamped * either.to(self.b_clamped.dtype)

    def time_bias(self, is_timed_q: Tensor, is_timed_k: Tensor,
                  is_clamped_q: Tensor | None = None,
                  is_clamped_k: Tensor | None = None) -> Tensor:
        """Both additive time flags at once — what attention layers should call."""
        b = self.untimed_bias(is_timed_q, is_timed_k)
        if is_clamped_q is not None and is_clamped_k is not None:
            b = b + self.clamped_bias(is_clamped_q, is_clamped_k)
        return b

    def extra_repr(self) -> str:
        w = self.omega
        return f"n_freq={self.n_freq}, omega=[{float(w[0]):g}, {float(w[-1]):g}] (fixed, non-persistent)"
