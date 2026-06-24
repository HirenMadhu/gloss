"""Leakage: no context cell may carry a timestamp after its seed's time."""
from __future__ import annotations

from gloss.data.collate import to_cell_batch

from .conftest import ENTITY, make_synth_batch, rel_f1_available, synthetic_bundle


def _n_leaks(cb) -> int:
    st = cb.seed_time.unsqueeze(1)
    return int(((cb.row_time > st) & cb.is_timed & ~cb.is_padding).sum())


def test_no_leakage_synth():
    cb = to_cell_batch(make_synth_batch(seed_time=100.0), synthetic_bundle(), ENTITY,
                       seq_len=16, max_fk=2)
    assert _n_leaks(cb) == 0


def test_planted_leak_is_detectable():
    # an event at t=150 > seed 100: its cells must be flagged by the leakage predicate
    cb = to_cell_batch(make_synth_batch(seed_time=100.0, event_times=(10.0, 150.0, 30.0, 40.0)),
                       synthetic_bundle(), ENTITY, seq_len=16, max_fk=2)
    assert _n_leaks(cb) >= 1


@rel_f1_available
def test_no_leakage_rel_f1():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=[8, 8], batch_size=16)
    cb = to_cell_batch(next(iter(loader)), bundle, task.entity_table, seq_len=512, max_fk=5)
    assert _n_leaks(cb) == 0
