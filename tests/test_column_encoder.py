"""Phase 2 — doc-conditioned node encoder: shapes, padding, and the FiLM regime response.

Uses cached rel-f1 for real TensorFrames (offline) + a HashEncoder grounding so the test is fast and
deterministic. The key assertion is that switching the doc regime full<->null *changes* node features
(the FiLM path actually conditions on documentation).
"""
from __future__ import annotations

import torch

from tests.conftest import rel_f1_available

D_TEXT = 64
D_MODEL = 32


def _setup():
    from relbench.tasks import get_task

    from gloss.data.collate import to_gloss_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.docs.cache import HashEncoder
    from gloss.docs.corpus import DocCorpus, schema_elements_from_db
    from gloss.docs.grounding import GroundingConfig, ground
    from gloss.model.column_encoder import ColumnEncoder

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=[6, 6], batch_size=4, shuffle=False)
    gb = to_gloss_batch(next(iter(loader)), bundle, task.entity_table, max_nodes=4096)

    from pathlib import Path

    corpus = DocCorpus.load(Path(__file__).resolve().parents[1] / "doc_corpus", "rel-f1")
    db = bundle.data  # not used; load db for elements
    from relbench.datasets import get_dataset

    db = get_dataset("rel-f1", download=False).get_db(upto_test_timestamp=False)
    elements = schema_elements_from_db(db)
    enc = HashEncoder(dim=D_TEXT)
    spans = corpus.spans(3)
    # threshold 0.0 so 'full' grounds every column (random docs) -> distinct from 'null'
    g_full = ground(elements, spans, enc, GroundingConfig(sim_threshold=0.0), regime="full")
    g_null = ground(elements, spans, enc, regime="null")

    torch.manual_seed(0)
    model = ColumnEncoder(bundle, d_model=D_MODEL, d_text=D_TEXT, n_freq=8)
    model.eval()
    return model, gb, g_full, g_null


@rel_f1_available
def test_node_states_shape_and_finite():
    model, gb, g_full, _ = _setup()
    with torch.no_grad():
        h = model(gb, g_full)
    assert h.shape == (gb.num_seeds, gb.n_max, D_MODEL)
    assert torch.isfinite(h).all()
    # pad positions are exactly zero
    pad = ~gb.pad_mask
    assert torch.count_nonzero(h[pad]) == 0


@rel_f1_available
def test_film_responds_to_doc_regime():
    model, gb, g_full, g_null = _setup()
    with torch.no_grad():
        h_full = model(gb, g_full)
        h_null = model(gb, g_null)
    # documentation must change the representation of real nodes
    diff = (h_full - h_null)[gb.pad_mask].abs().max().item()
    assert diff > 1e-4, "FiLM did not respond to the doc regime (full vs null)"
