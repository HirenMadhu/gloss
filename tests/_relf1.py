"""Cached builders for the rel-f1-guarded tests (build the graph once per session)."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def bundle_and_task():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    return bundle, task


@functools.lru_cache(maxsize=1)
def name_table(d_text: int = 64):
    """Frozen ``[C, d_text]`` column-name table for rel-f1 via a HashEncoder (no model download)."""
    from gloss.train.finetune import name_embeddings

    bundle, _task = bundle_and_task()
    return name_embeddings(bundle, "rel-f1", encoder="hash", d_text=d_text)


def sample_cell_batch(seq_len: int = 384, batch_size: int = 8, num_neighbors=(6, 6)):
    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import make_loader

    bundle, task = bundle_and_task()
    loader = make_loader(bundle, task, "train", num_neighbors=list(num_neighbors),
                         batch_size=batch_size, shuffle=False)
    raw = next(iter(loader))
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=seq_len, max_fk=5)
    return bundle, task, cb


def sample_join_batch(wide_len: int = 64, set_size: int = 32, batch_size: int = 16, fanout: int = 8):
    from gloss.setjoin.collate import to_join_batch
    from gloss.setjoin.paths import setjoin_neighbors

    from gloss.data.graph import make_loader

    bundle, task = bundle_and_task()
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout),
                         batch_size=batch_size, shuffle=False)
    raw = next(iter(loader))
    jb = to_join_batch(raw, bundle, task.entity_table, wide_len=wide_len, set_size=set_size)
    return bundle, task, jb


def sample_row_set_batch(m_rows: int = 32, cells_per_row: int = 48, batch_size: int = 8,
                         fanout: int = 8):
    from gloss.setjoin.collate import to_row_set_batch
    from gloss.setjoin.paths import setjoin_neighbors

    from gloss.data.graph import make_loader

    bundle, task = bundle_and_task()
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout),
                         batch_size=batch_size, shuffle=False)
    raw = next(iter(loader))
    rb = to_row_set_batch(raw, bundle, task.entity_table, m_rows=m_rows, cells_per_row=cells_per_row)
    return bundle, task, rb
