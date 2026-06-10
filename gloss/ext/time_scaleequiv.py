"""DEFERRED (Paper #2, Appendix E): scale-equivariant content-addressed temporal kernel. DO NOT BUILD.

    phi(dt)_k = sqrt(2/D) cos(w_k log(dt+eps) + b_k)                     # Bochner log-dt (scale-equivariant)
    (w_m, mu_m, raw_sigma_m) = MLP([h_i ; h_j]); sigma_m = softplus(.) + floor
    B_time(i,j) = sum_m w_m exp( -(log(dt+eps)-mu_m)^2 / (2 sigma_m^2) ) + b_head   # Hawkes-style mixture

~45-60% scoop-exposed; ship AFTER the measurement paper. ext_temporal=true only for this.
"""
from __future__ import annotations


class ScaleEquivariantTemporalKernel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DEFERRED to Paper #2 (Appendix E). Do not build in cycle 1.")
