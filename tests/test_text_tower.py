"""Phase-4+ — the depth-matched text tower + doc cross-attention.

Hermetic unit tests for the tower/cross-attn modules; guarded rel-f1 tests for the integrated two-tower
HALOS (full vs null responds to docs; scale-equivariance still holds because docs are time-independent).
"""
from __future__ import annotations

import torch

from gloss.model.text_tower import DocCrossAttention, TextTower
from tests.conftest import rel_f1_available


def test_text_tower_shapes_and_depth():
    tower = TextTower(d_text=16, d_model=32, n_layers=3, n_heads=4)
    states = tower(torch.randn(5, 16))
    assert len(states) == 3 and all(s.shape == (5, 32) for s in states)
    assert torch.isfinite(states[-1]).all()


def test_text_tower_empty_memory_is_empty_states():
    tower = TextTower(d_text=16, d_model=32, n_layers=2, n_heads=4)
    states = tower(torch.zeros(0, 16))           # null regime: empty span memory
    assert len(states) == 2 and all(s.shape == (0, 32) for s in states)


def test_doc_cross_attention_shapes_and_empty():
    ca = DocCrossAttention(d_model=32, n_heads=4)
    q = torch.randn(2, 7, 32)
    out = ca(q, torch.randn(5, 32))
    assert out.shape == (2, 7, 32) and torch.isfinite(out).all()
    # empty memory -> exactly zero contribution
    assert torch.count_nonzero(ca(q, torch.zeros(0, 32))) == 0


def _setup(feature, geometry, regime):
    from relbench.tasks import get_task

    from gloss.data.collate import to_gloss_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.model.halos import HALOS
    from gloss.train.finetune import docs_for_regime

    torch.manual_seed(0)
    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    gb = to_gloss_batch(next(iter(make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=4))),
                        bundle, task.entity_table)
    g, doc_mp = docs_for_regime(bundle, "rel-f1", regime, encoder="hash", d_text=32)
    model = HALOS(bundle, d_model=32, n_heads=4, n_layers=2, d_text=32, n_freq=8,
                  doc_cross_attn_feature=feature, doc_cross_attn_geometry=geometry).eval()
    return model, gb, g, doc_mp, bundle, task


@rel_f1_available
def test_two_tower_forward_and_doc_response():
    model, gb, gfull, doc_mp, bundle, task = _setup(feature=True, geometry=True, regime="full")
    from gloss.train.finetune import docs_for_regime

    gnull, _ = docs_for_regime(bundle, "rel-f1", "null", encoder="hash", d_text=32)
    with torch.no_grad():
        lf = model(gb, gfull, doc_per_metapath=doc_mp)
        ln = model(gb, gnull, doc_per_metapath=None)
    assert lf.shape == (gb.num_seeds, 1) and torch.isfinite(lf).all()
    assert (lf - ln).abs().max() > 1e-4, "cross-attn must respond to docs (full vs null)"
    assert len(model.text_tower.blocks) == 2     # depth-matched to n_layers


@rel_f1_available
def test_two_tower_preserves_scale_equivariance():
    from gloss.data.collate import to_gloss_batch
    from gloss.data.graph import make_loader

    model, _gb, g, doc_mp, bundle, task = _setup(feature=True, geometry=True, regime="full")
    raw = next(iter(make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=4)))

    def logits(c):
        r = raw.clone()
        for nt in r.node_types:
            if "time" in r[nt]:
                r[nt].time = r[nt].time.double() * c
            if nt == task.entity_table and "seed_time" in r[nt]:
                r[nt].seed_time = r[nt].seed_time.double() * c
        gb = to_gloss_batch(r, bundle, task.entity_table)
        with torch.no_grad():
            return model(gb, g, doc_per_metapath=doc_mp)

    base = logits(1.0)
    for c in (0.01, 100.0):
        assert torch.allclose(base, logits(c), atol=1e-5), f"cross-attn broke invariance at c={c}"
