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

**The ladder.** ``n_freq = 8`` frequencies ``ω_k``, log-spaced over ``[ω_min, ω_max] = [0.05, 5.0]``,
held in a **non-persistent buffer, never an** ``nn.Parameter`` **and never learned**:

    θ_r^(k) = ω_k · τ_r

The lowest frequency has period ``2π/0.05 ≈ 126`` in τ units, so it is monotonic across the whole 0–22
range with no wraparound; the highest has period ``≈ 1.26``, resolving recency *ratios* of about
``e^{1.26} ≈ 3.5×``.

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
        omega: ``(ω_min, ω_max)``, log-spaced inclusive band (default ``(0.05, 5.0)``).

    Attributes:
        omega: ``[n_freq]`` **non-persistent buffer** of fixed constants — not a parameter, absent from
            ``state_dict()``, byte-identical across any two models regardless of the data they saw.
        b_untimed: the single learned scalar added to attention logits touching an untimed endpoint.
    """

    def __init__(self, n_freq: int = 8, omega: tuple[float, float] = (0.05, 5.0)):
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

    def feats(self, tau: Tensor, is_timed: Tensor | None = None) -> Tensor:
        """``[sin θ ; cos θ] ∈ R^{2·n_freq}`` -> ``[..., 2*n_freq]``, **all-zero** where untimed.

        Zeroing (rather than leaving ``cos 0 = 1``) is what keeps "untimed" linearly distinguishable
        from "Δ = 0" downstream of the learned ``W_τ`` of §3.3.
        """
        th = self.theta(tau)                                    # unmasked: mask the pair, not θ
        f = torch.cat((torch.sin(th), torch.cos(th)), dim=-1)   # [..., 2*n_freq]
        if is_timed is not None:
            f = f * is_timed.unsqueeze(-1).to(f.dtype)
        return f

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

    def extra_repr(self) -> str:
        w = self.omega
        return f"n_freq={self.n_freq}, omega=[{float(w[0]):g}, {float(w[-1]):g}] (fixed, non-persistent)"
