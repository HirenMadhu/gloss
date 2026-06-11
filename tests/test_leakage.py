"""Phase 0 — the headline invariant: no context row has row_time > seed_time.

Checked on the synthetic fixture (controlled) and, when rel-f1 is cached, on a real sampled batch
(the relbench temporal sampler must never leak the future into a seed's subgraph).
"""
from __future__ import annotations

import torch

from gloss.data.collate import to_gloss_batch
from tests.conftest import make_dualfk_batch, rel_f1_available


def _violations(gb) -> int:
    # timestamped node row_time must be <= its segment's seed_time
    return int((gb.is_timed & (gb.row_time > gb.seed_time.view(-1, 1))).sum())


def test_synthetic_leakfree(dualfk_bundle):
    gb = to_gloss_batch(make_dualfk_batch(seed_time=100.0), dualfk_bundle, "user", max_nodes=64)
    assert _violations(gb) == 0


def test_synthetic_detects_planted_leak(dualfk_bundle):
    # plant an event AFTER the seed time; the *check* must catch it (guards the assertion itself)
    leaky = make_dualfk_batch(seed_time=25.0, event_times=(10.0, 20.0, 30.0, 40.0))
    gb = to_gloss_batch(leaky, dualfk_bundle, "user", max_nodes=64)
    assert _violations(gb) > 0


@rel_f1_available
def test_real_rel_f1_sampler_is_leakfree():
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=[10, 10], batch_size=16, shuffle=False)
    raw = next(iter(loader))
    gb = to_gloss_batch(raw, bundle, task.entity_table, max_nodes=4096)
    assert _violations(gb) == 0
    # also: every temporal_valid pair links two real, timed nodes
    tv = gb.temporal_valid
    assert bool((tv <= (gb.is_timed.unsqueeze(2) & gb.is_timed.unsqueeze(1))).all())
