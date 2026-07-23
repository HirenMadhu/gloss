"""RowModel gate bookkeeping (hermetic): the rowmodel-* variant labels (collision fix) + record
round-trip. The 27-cell grid, NMAE math, and compare_table are shared with SetJoin (test_setjoin_runner)."""
from __future__ import annotations

import math

import gloss.eval.ablation as ablation
from gloss.eval.ablation import LEADERBOARD_TASKS, aggregate, format_table
from gloss.setjoin.runner import gate_grid, variant_of


def test_rowmodel_grid_is_27(monkeypatch):
    monkeypatch.setattr(ablation, "entity_tasks", lambda ds: sorted(LEADERBOARD_TASKS.get(ds, ())))
    assert len(gate_grid(seeds=3)) == 27           # the grid is model-agnostic (9 tasks x 3 seeds)


def test_rowmodel_variant_labels_and_no_collision():
    v = lambda **mk: variant_of(mk, model="rowmodel")
    # default arm
    assert v(route_on="signature") == "rowmodel-signature"
    assert variant_of(None, model="rowmodel") == "rowmodel-signature"
    assert v(route_on="dense") == "rowmodel-dense"          # rowmodel keeps the route arm in the label
    # NON-default axes each get a distinct suffix so {index}_{variant}.json can't collide
    assert v(route_on="signature", aggregate="slot") == "rowmodel-signature-aggslot"
    assert v(route_on="signature", use_counts=True) == "rowmodel-signature-counts"
    assert v(route_on="signature", row_pool="gated") == "rowmodel-signature-rpgated"
    assert v(route_on="signature", cell_slots=(4,)) == "rowmodel-signature-cs4"
    assert v(route_on="signature", cell_slots=(16, 4, 2)) == "rowmodel-signature-cs16x4x2"
    assert v(route_on="signature", cell_slots=(8, 2)) == "rowmodel-signature"   # default -> no suffix
    # the corruption the fix prevents: mean vs slot (or gated) must NOT share a filename
    labels = {v(route_on="signature"), v(route_on="signature", aggregate="slot"),
              v(route_on="signature", row_pool="gated"), v(route_on="signature", use_counts=True)}
    assert len(labels) == 4
    # rowmodel and setjoin never collide either
    assert v(route_on="signature") != variant_of(dict(route_on="signature"), model="setjoin")


def test_rowmodel_records_round_trip():
    recs = [{"dataset": "rel-f1", "task": "driver-dnf", "signal": "setjoin",
             "variant": "rowmodel-signature", "seed": s, "task_type": "binary",
             "test_roc_auc": 0.80 + 0.02 * s} for s in range(3)]
    agg = aggregate(recs, keys=("test_roc_auc",))
    mean, _sd, _ci, n = agg[("rel-f1", "driver-dnf", "rowmodel-signature")]["test_roc_auc"]
    assert n == 3 and math.isclose(mean, 0.82)
    out = format_table(recs, split="test", baseline="rowmodel-signature")
    assert "rowmodel-signature" in out and "rel-f1 / driver-dnf" in out
