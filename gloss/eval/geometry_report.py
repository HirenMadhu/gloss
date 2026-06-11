"""Phase 3 — the readable-geometry exhibit.

Because the geometry is *compiled from documentation per relation*, we can read off exactly what the
model decided for each FK role / metapath: the per-head Gaussian-in-τ kernel ``(a, μ, σ, b)``. This is
the figure that sells the method ("links through `placed_by` matter at τ≈recency; links through
`renewal` matter at τ≈+5.9 (~1 year)"). Works even on an untrained model — it's a pipeline check here.
"""
from __future__ import annotations

import torch

from ..data.graph import MP_MULTIHOP, MP_PAD, MP_SELF, GraphBundle


def metapath_label(bundle: GraphBundle, mp_id: int) -> str:
    if mp_id == MP_PAD:
        return "<pad>"
    if mp_id == MP_SELF:
        return "<self>"
    if mp_id == MP_MULTIHOP:
        return "<multihop>"
    for col, i in bundle.metapath_id.items():
        if i == mp_id:
            return f"fk:{col}"
    return f"mp:{mp_id}"


def geometry_report(model, bundle: GraphBundle, *, doc_per_metapath=None) -> dict:
    """Return per-metapath, per-head compiled kernel params as plain Python (for printing/plotting)."""
    model.eval()
    with torch.no_grad():
        geom = model.compile_geometry(doc_per_metapath)
    rows = []
    for mp_id in range(geom.num_metapaths):
        rows.append({
            "metapath_id": mp_id,
            "label": metapath_label(bundle, mp_id),
            "a": geom.a[mp_id].tolist(),
            "mu": geom.mu[mp_id].tolist(),
            "sigma": geom.sigma[mp_id].tolist(),
            "b": geom.b[mp_id].tolist(),
        })
    return {"n_heads": geom.n_heads, "num_metapaths": geom.num_metapaths, "kernels": rows}


def format_report(report: dict, *, head: int = 0) -> str:
    lines = [f"compiled geometry — {report['num_metapaths']} metapaths x {report['n_heads']} heads "
             f"(showing head {head}):",
             f"  {'metapath':16s} {'a':>8s} {'mu':>8s} {'sigma':>8s} {'b':>8s}"]
    for r in report["kernels"]:
        lines.append(f"  {r['label']:16s} {r['a'][head]:8.3f} {r['mu'][head]:8.3f} "
                     f"{r['sigma'][head]:8.3f} {r['b'][head]:8.3f}")
    return "\n".join(lines)
