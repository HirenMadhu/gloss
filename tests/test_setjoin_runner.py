"""SetJoin gate bookkeeping (hermetic: grid, NMAE, aggregate round-trip, compare table — no training)."""
from __future__ import annotations

import math

import gloss.eval.ablation as ablation
from gloss.eval.ablation import LEADERBOARD_TASKS, aggregate, format_table
from gloss.setjoin.runner import (GATE_DATASETS, attach_nmae, compare_table, diff_table,
                                  gate_grid, variant_of)


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


def test_variant_of_labels_all_arms():
    assert variant_of(dict(route_on="dense")) == "setjoin"
    assert variant_of(dict(route_on="signature")) == "setjoin-signature"
    assert variant_of(dict(route_on="signature", use_shared=True)) == "setjoin-signature-shared"
    assert variant_of(dict(route_on="hidden", use_shared=True)) == "setjoin-hidden-shared"
    assert variant_of(dict(route_on="dense", use_shared=True)) == "setjoin"   # dense has no MoE
    assert variant_of(None) == "setjoin-signature"
    # hop-2 / per-relation-cap suffixes (defaults leave every label above unchanged)
    assert variant_of(dict(route_on="signature"), fanout2=8) == "setjoin-signature-hop2"
    assert variant_of(dict(route_on="signature"), per_relation_cap=32) == "setjoin-signature-cap32"
    assert variant_of(dict(route_on="signature"), fanout2=8,
                      per_relation_cap=32) == "setjoin-signature-hop2-cap32"
    assert variant_of(dict(route_on="dense"), fanout2=8) == "setjoin-hop2"


def test_variant_of_is_readout_aware():
    # readout=pma keeps the current label (back-compat); measure/slot get distinct suffixes so the
    # {index}_{variant}.json gate files can't collide and silently reuse another arm's record.
    assert variant_of(dict(route_on="signature", readout="pma")) == "setjoin-signature"
    assert variant_of(dict(route_on="signature")) == "setjoin-signature"          # default = pma
    assert variant_of(dict(route_on="signature", readout="measure")) == "setjoin-signature-measure"
    assert variant_of(dict(route_on="signature", readout="slot",
                           slot_mode="hard")) == "setjoin-signature-slothard"
    assert variant_of(dict(route_on="signature", readout="slot",
                           slot_mode="soft")) == "setjoin-signature-slotsoft"
    assert (variant_of(dict(route_on="signature", readout="slot", slot_mode="hard"))
            != variant_of(dict(route_on="signature", readout="pma")))


def test_diff_table_deltas():
    a = [_rec("rel-f1", "driver-dnf", s, task_type="binary", test_roc_auc=0.82) for s in range(3)]
    a += [_rec("rel-f1", "driver-position", s, task_type="regression", test_nmae=0.40)
          for s in range(3)]
    b = [_rec("rel-f1", "driver-dnf", s, task_type="binary", test_roc_auc=0.80) for s in range(3)]
    b += [_rec("rel-f1", "driver-position", s, task_type="regression", test_nmae=0.45)
          for s in range(3)]
    out = diff_table(a, b, label_a="hop2", label_b="ctrl")
    assert "driver-dnf" in out and "Δ= +2.000" in out.replace("Δ=+", "Δ= +")
    assert "driver-position" in out and "-0.050" in out


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
    assert "single table" in out and "multi table" in out and "RT" in out and "GelGT" in out
    assert "±" not in out                                  # no CI/std in the table
    assert "\033[1m" in out and "\033[4m" in out           # best bolded, 2nd best underlined
    # driver-dnf: single table 81.0 > RT 78.7 > GelGT 76.1, MoRE 82.9 → best = multi table (82.9 bold),
    # 2nd = single table (81.0 underlined)
    assert "\033[1m" + f"{82.9:>12.3f}" + "\033[0m" in out
    assert "\033[4m" + f"{81.0:>14.3f}" + "\033[0m" in out
    assert "beats RT" in out


def test_compare_table_empty():
    assert compare_table([]) == "(no records)"
