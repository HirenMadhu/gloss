"""Phase 3 — the headline invariant: a global clock rescale t -> c*t leaves the model's logits
*identical* (default config), because time enters only through the dimensionless tau. With
``absolute_anchor=True`` the geometry is pinned to an absolute scale and the logits MUST change
(the test bites both ways).

Uses a real rel-f1 batch (guarded by cache); times are rescaled at the raw-batch level and the dense
batch is recomputed, so tau/dt/T_ctx are regenerated exactly as in training.
"""
from __future__ import annotations

import torch

from tests.conftest import rel_f1_available

D_TEXT, D_MODEL, H = 32, 32, 4


def _rescale_raw(raw, entity_table, c: float):
    """Multiply every row timestamp + seed time by c (float64). Clone so we never mutate the shared raw."""
    r = raw.clone()
    for nt in r.node_types:
        store = r[nt]
        if "time" in store:
            store.time = store.time.double() * c
        if nt == entity_table and "seed_time" in store:
            store.seed_time = store.seed_time.double() * c
    return r


def _build(absolute_anchor: bool):
    from pathlib import Path

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from gloss.data.collate import to_gloss_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.docs.cache import HashEncoder
    from gloss.docs.corpus import DocCorpus, schema_elements_from_db
    from gloss.docs.grounding import GroundingConfig, ground
    from gloss.model.halos import HALOS, build_doc_per_metapath

    torch.manual_seed(0)
    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=4, shuffle=False)
    raw = next(iter(loader))

    db = get_dataset("rel-f1", download=False).get_db(upto_test_timestamp=False)
    els = schema_elements_from_db(db)
    corpus = DocCorpus.load(Path(__file__).resolve().parents[1] / "doc_corpus", "rel-f1")
    g = ground(els, corpus.spans(3), HashEncoder(D_TEXT), GroundingConfig(sim_threshold=0.0), regime="full")
    doc_mp = build_doc_per_metapath(bundle, g)

    model = HALOS(bundle, d_model=D_MODEL, n_heads=H, n_layers=2, d_text=D_TEXT, n_freq=8,
                  absolute_anchor=absolute_anchor).eval()

    def logits_for(c):
        rb = raw if c == 1.0 else _rescale_raw(raw, task.entity_table, c)
        gb = to_gloss_batch(rb, bundle, task.entity_table, max_nodes=4096)
        with torch.no_grad():
            return model(gb, g, doc_per_metapath=doc_mp)

    return logits_for


@rel_f1_available
def test_logits_invariant_under_time_rescale():
    logits_for = _build(absolute_anchor=False)
    base = logits_for(1.0)
    for c in (0.01, 7.3, 100.0):
        assert torch.allclose(base, logits_for(c), atol=1e-5), f"not invariant at c={c}"


def test_absolute_anchor_breaks_invariance(dualfk_batch, dualfk_bundle):
    """Hermetic: with the absolute anchor wired in, feeding different log T_ctx (which is exactly what a
    clock rescale changes) moves the geometry — so the model is deliberately NOT scale-invariant. Tested
    at the attention level with an active Gaussian so the effect is guaranteed regardless of init."""
    from gloss.data.collate import to_gloss_batch
    from gloss.model.attention import RelationalAttention
    from gloss.model.bias_generator import GeometryTable

    gb = to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)
    assert gb.temporal_valid.any(), "fixture must have temporal pairs for this test to mean anything"

    P, H = dualfk_bundle.num_metapaths, 4
    geom = GeometryTable(                # active Gaussian (a=1, mu=0, sigma=1) + anchor coefficient 1
        a=torch.ones(P, H), mu=torch.zeros(P, H), sigma=torch.ones(P, H),
        b=torch.zeros(P, H), anchor_w=torch.ones(P, H),
    )
    torch.manual_seed(0)
    attn = RelationalAttention(16, H).eval()
    h = torch.randn(gb.num_seeds, gb.n_max, 16)
    with torch.no_grad():
        out0 = attn(h, gb, geom, log_t_ctx=torch.zeros(gb.num_seeds))
        out3 = attn(h, gb, geom, log_t_ctx=torch.full((gb.num_seeds,), 3.0))
    assert not torch.allclose(out0[gb.pad_mask], out3[gb.pad_mask]), "absolute anchor must break invariance"
