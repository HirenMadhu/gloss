"""Phase 3 — the core operator: compile-once, sigma floor, doc-conditioning, kernel diversity."""
from __future__ import annotations

import torch

from gloss.model.bias_generator import BiasGenerator

P, H, DT = 7, 4, 8


def test_compile_shapes_and_sigma_floor():
    torch.manual_seed(0)
    bg = BiasGenerator(P, H, d_text=DT, sigma_floor=0.25)
    geom = bg.compile()
    for t in (geom.a, geom.mu, geom.sigma, geom.b):
        assert t.shape == (P, H) and torch.isfinite(t).all()
    assert (geom.sigma >= 0.25 - 1e-6).all()
    assert geom.anchor_w is None


def test_compile_is_deterministic():
    torch.manual_seed(0)
    bg = BiasGenerator(P, H, d_text=DT)
    a, b = bg.compile(), bg.compile()
    assert torch.equal(a.a, b.a) and torch.equal(a.mu, b.mu) and torch.equal(a.sigma, b.sigma)


def test_kernel_diversity_across_metapaths():
    torch.manual_seed(1)
    bg = BiasGenerator(P, H, d_text=DT)
    geom = bg.compile()
    # not all metapath rows are identical (distinct E_metapath -> distinct geometry)
    rows = torch.stack([geom.a, geom.mu, geom.sigma, geom.b], dim=-1)  # [P,H,4]
    assert not torch.allclose(rows[0], rows[1])


def test_geometry_is_doc_conditioned():
    torch.manual_seed(2)
    bg = BiasGenerator(P, H, d_text=DT)
    docA = torch.zeros(P, DT)
    docB = torch.zeros(P, DT)
    docB[3] = torch.randn(DT)              # change only metapath 3's documentation
    gA, gB = bg.compile(docA), bg.compile(docB)
    # the changed row's geometry moves; an unchanged row does not
    assert not torch.allclose(gA.a[3], gB.a[3])
    assert torch.allclose(gA.a[4], gB.a[4])


def test_absolute_anchor_emits_extra_coefficient():
    bg = BiasGenerator(P, H, d_text=DT, absolute_anchor=True)
    geom = bg.compile()
    assert geom.anchor_w is not None and geom.anchor_w.shape == (P, H)
