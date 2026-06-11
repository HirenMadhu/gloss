"""Phase 3 — THE CORE OPERATOR: attention geometry generated from documentation.

For each typed relation / metapath ``p`` we generate, per head ``h``, the parameters of a Gaussian-in-τ
attention bias from a context vector built out of *documentation + typed structure*:

    ctx(p)              = [ E_metapath(p) ;  doc(p) ]            # doc(p) = pooled FK-role documentation
    (a, μ, σ_raw, b)    = g_θ(ctx(p))                            # small MLP, per head
    σ                   = softplus(σ_raw) + σ_floor

The table is **compiled once per DB** over the (small) set of metapath ids — `compile()` runs g_θ over
every id and returns a :class:`GeometryTable`. It is recomputed each forward (cheap; gradients flow into
g_θ), *not* a frozen cache. Because ``ctx(p)`` depends only on docs + structure (never node content),
the per-token cost is zero — the same table is broadcast to all pairs.

Two FKs into one table get different metapath ids → different ``ctx`` → different generated geometry:
this is how HALOS disambiguates dual foreign keys (a thing RT cannot do).

Flags (off by default):
* ``absolute_anchor`` — also emit a per-head coefficient on ``log T_ctx`` so the Gaussian centre can be
  pinned to an *absolute* timescale. This deliberately **breaks** strict scale-equivariance (the
  scale-equivariance test asserts it bites), letting docs express "annual"-style absolute intents.
* ``content_modulated`` — (v2 ablation, not generated here) a residual Δμ,Δσ from ``[h_u;h_w]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class GeometryTable:
    a: Tensor          # [P, H]   amplitude
    mu: Tensor         # [P, H]   Gaussian centre in τ
    sigma: Tensor      # [P, H]   Gaussian width (>= sigma_floor)
    b: Tensor          # [P, H]   structural (constant) bias
    anchor_w: Tensor | None  # [P, H] or None; coefficient on log T_ctx when absolute_anchor

    @property
    def num_metapaths(self) -> int:
        return self.a.shape[0]

    @property
    def n_heads(self) -> int:
        return self.a.shape[1]


class BiasGenerator(nn.Module):
    def __init__(
        self,
        num_metapaths: int,
        n_heads: int,
        *,
        d_text: int = 2560,
        d_meta: int = 32,
        hidden: int = 128,
        sigma_floor: float = 0.1,
        absolute_anchor: bool = False,
    ):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.n_heads = n_heads
        self.sigma_floor = sigma_floor
        self.absolute_anchor = absolute_anchor
        self.n_out = 5 if absolute_anchor else 4

        self.e_metapath = nn.Embedding(num_metapaths, d_meta)
        self.null_doc = nn.Parameter(torch.zeros(d_text))
        self.mlp = nn.Sequential(
            nn.Linear(d_meta + d_text, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads * self.n_out),
        )

    def compile(self, doc_per_metapath: Tensor | None = None) -> GeometryTable:
        """Run g_θ over all metapath ids -> a GeometryTable. ``doc_per_metapath`` is ``[P, d_text]``
        (pooled FK-role docs); rows left as None fall back to the learned ``null_doc``."""
        P = self.num_metapaths
        ids = torch.arange(P, device=self.e_metapath.weight.device)
        e = self.e_metapath(ids)                                  # [P, d_meta]
        if doc_per_metapath is None:
            doc = self.null_doc.unsqueeze(0).expand(P, -1)
        else:
            doc = doc_per_metapath.to(e.dtype).to(e.device)
        out = self.mlp(torch.cat([e, doc], dim=-1))               # [P, H*n_out]
        out = out.view(P, self.n_heads, self.n_out)
        a, mu, sig_raw, b = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
        sigma = F.softplus(sig_raw) + self.sigma_floor
        anchor_w = out[..., 4] if self.absolute_anchor else None
        return GeometryTable(a=a, mu=mu, sigma=sigma, b=b, anchor_w=anchor_w)
