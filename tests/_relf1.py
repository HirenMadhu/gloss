"""Cached builders for the rel-f1-guarded tests (build the graph once per session)."""
from __future__ import annotations

import functools
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def bundle_and_task():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    return bundle, task


@functools.lru_cache(maxsize=1)
def groundings(d_text: int = 64):
    """(full, null, name_only) groundings from the rel-f1 corpus via a HashEncoder (no model download).
    sim_threshold=0 so `full` grounds every column (the FiLM signal is exercised)."""
    from relbench.datasets import get_dataset

    from gloss.docs.cache import HashEncoder
    from gloss.docs.corpus import DocCorpus, schema_elements_from_db
    from gloss.docs.grounding import GroundingConfig, ground

    corpus = DocCorpus.load(REPO / "doc_corpus", "rel-f1")
    db = get_dataset("rel-f1", download=False).get_db(upto_test_timestamp=False)
    elements = schema_elements_from_db(db)
    spans = corpus.spans(3)
    enc = HashEncoder(dim=d_text)
    cfg = GroundingConfig(sim_threshold=0.0)
    return (
        ground(elements, spans, enc, cfg, regime="full"),
        ground(elements, spans, enc, cfg, regime="null"),
        ground(elements, spans, enc, cfg, regime="name_only"),
    )


def sample_cell_batch(seq_len: int = 384, batch_size: int = 8, num_neighbors=(6, 6)):
    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import make_loader

    bundle, task = bundle_and_task()
    loader = make_loader(bundle, task, "train", num_neighbors=list(num_neighbors),
                         batch_size=batch_size, shuffle=False)
    raw = next(iter(loader))
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=seq_len, max_fk=5)
    return bundle, task, cb
