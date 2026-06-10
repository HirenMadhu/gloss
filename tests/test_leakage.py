"""test_leakage.py — THE core invariant: no context cell has ``row_time > seed_time`` (impl §9, §4.2)."""
from __future__ import annotations

import pytest

from gloss.data.relbench_graph import MASK_NEIGHBOR
from gloss.data.synthetic import make_synthetic_bundle


def _all_rows_respect_leakage(bundle, sampler, n_seeds=60, seed=0):
    seeds = bundle.task.get_table("train").df
    seeds = seeds.sample(n=min(n_seeds, len(seeds)), random_state=seed)
    checked, timed_context = 0, 0
    for row in seeds.itertuples():
        sg = sampler.sample("entity", row.id, row.date, row.target)
        for r in sg.rows:
            # TIME_MIN (timeless static rows) trivially satisfies <= ; event rows must be in the past
            assert r.row_time_ns <= sg.seed_time_ns, (
                f"LEAKAGE: {r.table} row_time {r.row_time_ns} > seed_time {sg.seed_time_ns}")
            checked += 1
            if r.mask_kind == MASK_NEIGHBOR:  # a real timestamped event pulled into context
                timed_context += 1
    # guard against a vacuous pass: the invariant must be exercised on actual timestamped context rows
    return checked, timed_context


def test_synthetic_no_temporal_leakage():
    bundle, _ = make_synthetic_bundle(seed=0, num_entities=200)
    sampler = bundle.make_sampler(num_neighbors=[10, 10], max_cells=2048, seed=0)
    checked, timed_context = _all_rows_respect_leakage(bundle, sampler)
    assert checked > 0, "sampler returned no cells to check"
    assert timed_context > 0, "no timestamped event context sampled — leakage check would be vacuous"


def test_leakage_holds_across_fanouts_and_seeds():
    for s in (1, 2):
        bundle, _ = make_synthetic_bundle(seed=s, num_entities=150)
        for fanout in ([4], [8, 8], [6, 6, 6]):
            sampler = bundle.make_sampler(num_neighbors=fanout, max_cells=4096, seed=s)
            _all_rows_respect_leakage(bundle, sampler, n_seeds=25, seed=s)


@pytest.mark.relbench
def test_relf1_no_temporal_leakage():
    """Same invariant on real data (rel-f1). Skipped if the dataset isn't downloaded / no network."""
    rb = pytest.importorskip("relbench")
    from gloss.data.relbench_graph import load_task_bundle

    try:
        bundle = load_task_bundle("rel-f1", "driver-dnf", download=True)
    except Exception as e:  # network / cache miss
        pytest.skip(f"rel-f1 unavailable: {e}")
    sampler = bundle.make_sampler(num_neighbors=[8, 8], max_cells=4096, seed=0)
    seeds = bundle.task.get_table("train").df.sample(n=40, random_state=0)
    for row in seeds.itertuples(index=False):
        sg = sampler.sample(bundle.entity_table, getattr(row, bundle.entity_col),
                            getattr(row, bundle.time_col), getattr(row, bundle.target_col))
        for r in sg.rows:
            assert r.row_time_ns <= sg.seed_time_ns
