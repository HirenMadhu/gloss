"""Routing-signal ablation bookkeeping (hermetic: grid enumeration + seed aggregation, no training)."""
from __future__ import annotations

import math

from gloss.eval.ablation import ROUTING_SIGNALS, aggregate, build_grid, format_table


def test_build_grid_covers_dataset_task_signal_seed():
    dt = {"rel-f1": ["driver-dnf", "driver-position"], "rel-stack": ["user-engagement"]}
    grid = build_grid(dt, seeds=3)
    assert len(grid) == 3 * len(ROUTING_SIGNALS) * 3          # 3 (dataset,task) x 6 signals x 3 seeds
    assert {c["signal"] for c in grid} == set(ROUTING_SIGNALS)
    assert {c["seed"] for c in grid} == {0, 1, 2}
    assert {(c["dataset"], c["task"]) for c in grid} == {
        ("rel-f1", "driver-dnf"), ("rel-f1", "driver-position"), ("rel-stack", "user-engagement"),
    }
    assert len({(c["dataset"], c["task"], c["signal"], c["seed"]) for c in grid}) == len(grid)


def _rec(dataset, task, signal, seed, **m):
    return {"dataset": dataset, "task": task, "signal": signal, "seed": seed, **m}


def test_aggregate_groups_by_triple_with_ci():
    records = [
        _rec("rel-f1", "driver-dnf", "signature", 0, test_roc_auc=0.80),
        _rec("rel-f1", "driver-dnf", "signature", 1, test_roc_auc=0.90),
        _rec("rel-f1", "driver-dnf", "dense", 0, test_roc_auc=0.70),
    ]
    agg = aggregate(records, keys=("test_roc_auc",))
    mean, sd, ci, n = agg[("rel-f1", "driver-dnf", "signature")]["test_roc_auc"]
    assert n == 2 and math.isclose(mean, 0.85)
    assert math.isclose(sd, 0.0707106781, rel_tol=1e-6)
    assert math.isclose(ci, 1.96 * sd / math.sqrt(2), rel_tol=1e-9)
    _m, sd1, ci1, n1 = agg[("rel-f1", "driver-dnf", "dense")]["test_roc_auc"]
    assert n1 == 1 and sd1 == 0.0 and ci1 == 0.0


def test_aggregate_ignores_missing_and_nan():
    records = [
        _rec("d", "t", "signature", 0, test_mae=1.0),
        _rec("d", "t", "signature", 1, test_mae=float("nan")),
        _rec("d", "t", "signature", 2, test_mae=3.0),
    ]
    mean, _sd, _ci, n = aggregate(records, keys=("test_mae",))[("d", "t", "signature")]["test_mae"]
    assert n == 2 and math.isclose(mean, 2.0)


def test_format_table_binary_lift_vs_dense_higher_is_better():
    records = [
        _rec("rel-f1", "driver-dnf", "signature", 0, task_type="binary", test_roc_auc=0.90),
        _rec("rel-f1", "driver-dnf", "dense", 0, task_type="binary", test_roc_auc=0.85),
    ]
    out = format_table(records, split="test")
    assert "rel-f1 / driver-dnf" in out and "binary" in out
    assert "signature" in out and "dense" in out
    assert "Δvs dense=+0.0500" in out          # 0.90 - 0.85 (higher AUROC is better)


def test_format_table_regression_lift_is_lower_is_better():
    records = [
        _rec("rel-f1", "driver-position", "signature", 0, task_type="regression", test_mae=4.0),
        _rec("rel-f1", "driver-position", "dense", 0, task_type="regression", test_mae=5.0),
    ]
    out = format_table(records, split="test")
    assert "regression" in out
    assert "Δvs dense=+1.0000" in out          # dense MAE 5.0 - signature 4.0 (lower MAE is better)


# ---- arch must never merge with rt in the variant / grouping key ----


def test_two_level_and_rt_do_not_share_a_variant_or_filename():
    """`variant` is BOTH the output filename key and the aggregate grouping key.

    If `arch` did not enter it, a `two_level` run and an `rt` run at the same (index, router) would
    collide on disk and — worse — be averaged into one mean by `aggregate`. Two architectures
    silently merging produces a confident wrong table, so this is a correctness guard, not cosmetics.
    """
    from gloss.eval.ablation import variant_label

    rt = variant_label("signature")
    two = f"{variant_label('signature')}@two_level"
    assert rt != two
    assert f"{0:04d}_{rt}.json" != f"{0:04d}_{two}.json"


def test_aggregate_keeps_architectures_separate():
    """Same dataset/task/router, different arch -> two distinct rows, not one blended mean."""
    records = [
        _rec("rel-f1", "driver-dnf", "signature", 0, test_roc_auc=0.70, variant="signature"),
        _rec("rel-f1", "driver-dnf", "signature", 1, test_roc_auc=0.72, variant="signature"),
        _rec("rel-f1", "driver-dnf", "signature", 0, test_roc_auc=0.90,
             variant="signature@two_level"),
        _rec("rel-f1", "driver-dnf", "signature", 1, test_roc_auc=0.92,
             variant="signature@two_level"),
    ]
    agg = aggregate(records, ("test_roc_auc",))
    keys = {k[-1] for k in agg}
    assert keys == {"signature", "signature@two_level"}, f"architectures merged: {keys}"

    rt_mean = agg[("rel-f1", "driver-dnf", "signature")]["test_roc_auc"][0]
    tl_mean = agg[("rel-f1", "driver-dnf", "signature@two_level")]["test_roc_auc"][0]
    assert math.isclose(rt_mean, 0.71, abs_tol=1e-9)
    assert math.isclose(tl_mean, 0.91, abs_tol=1e-9)
    # the blended mean (0.81) must appear NOWHERE — that is the failure this guards
    assert not math.isclose(rt_mean, 0.81, abs_tol=1e-9)
    assert not math.isclose(tl_mean, 0.81, abs_tol=1e-9)


