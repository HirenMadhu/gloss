"""SetJoin gate bookkeeping (hermetic: grid, NMAE, aggregate round-trip, compare table — no training)."""
from __future__ import annotations

import math

import gloss.eval.ablation as ablation
from gloss.eval.ablation import LEADERBOARD_TASKS, aggregate, format_table
from gloss.setjoin.runner import GATE_DATASETS, attach_nmae, compare_table, gate_grid


def test_gate_grid_is_9_tasks_x_seeds(monkeypatch):
    # entity_tasks needs the relbench registry; pin it to the leaderboard sets (hermetic)
    monkeypatch.setattr(ablation, "entity_tasks",
                        lambda ds: sorted(LEADERBOARD_TASKS.get(ds, ())))
    grid = gate_grid(seeds=3)
    assert len(grid) == 27                                 # 9 leaderboard tasks x 3 seeds
    assert {c["signal"] for c in grid} == {"setjoin"}
    assert {c["dataset"] for c in grid} == set(GATE_DATASETS)
    assert {(c["dataset"], c["task"]) for c in grid} == {
        (ds, t) for ds, ts in LEADERBOARD_TASKS.items() for t in ts}
    assert len({tuple(sorted(c.items())) for c in grid}) == 27


def test_attach_nmae():
    rec = {"task_type": "regression", "test_mae": 3.0}
    assert math.isclose(attach_nmae(rec, 2.0)["test_nmae"], 1.5)
    assert "test_nmae" not in attach_nmae({"task_type": "binary", "test_mae": 3.0}, 2.0)
    assert "test_nmae" not in attach_nmae({"task_type": "regression"}, 2.0)


def _rec(ds, task, seed, **m):
    return {"dataset": ds, "task": task, "signal": "setjoin", "variant": "setjoin", "seed": seed, **m}


def test_records_round_trip_through_ablation_bookkeeping():
    records = [
        _rec("rel-f1", "driver-dnf", 0, task_type="binary", test_roc_auc=0.80),
        _rec("rel-f1", "driver-dnf", 1, task_type="binary", test_roc_auc=0.90),
    ]
    agg = aggregate(records, keys=("test_roc_auc",))
    mean, _sd, _ci, n = agg[("rel-f1", "driver-dnf", "setjoin")]["test_roc_auc"]
    assert n == 2 and math.isclose(mean, 0.85)
    out = format_table(records, split="test", baseline="setjoin")
    assert "rel-f1 / driver-dnf" in out and "setjoin" in out
    assert "Δvs setjoin=+0.0000" in out                    # single-arm study: Δ vs itself


def test_compare_table_renders_vs_leaderboard_baselines():
    records = [
        _rec("rel-f1", "driver-dnf", s, task_type="binary", test_roc_auc=0.80 + 0.01 * s)
        for s in range(3)
    ] + [
        _rec("rel-f1", "driver-position", s, task_type="regression",
             test_mae=1.6, test_nmae=0.40 + 0.01 * s)
        for s in range(3)
    ]
    out = compare_table(records)                           # reads the repo baselines JSON (offline)
    assert "rel-f1 / driver-dnf" in out and "rel-f1 / driver-position" in out
    assert "81.000" in out                                 # AUROC mean x100
    assert "0.410" in out                                  # NMAE mean
    assert "RT(scratch)" in out and "GelGT" in out and "MoRE best" in out
    assert "* = beats RT" in out


def test_compare_table_empty():
    assert compare_table([]) == "(no records)"
