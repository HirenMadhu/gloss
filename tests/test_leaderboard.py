"""The external-baseline loader: coverage of the 9 tasks, metric conventions, win direction."""

from __future__ import annotations

import json

import pytest

from gloss.eval.ablation import LEADERBOARD_TASKS
from gloss.eval.leaderboard import METHODS, beats, load, summary_line, to_display

ALL_KEYS = [f"{ds}/{tk}" for ds, tasks in LEADERBOARD_TASKS.items() for tk in tasks]


def test_covers_all_nine_tasks_for_both_methods():
    base = load()
    assert set(base) == set(ALL_KEYS), "every leaderboard task needs an external baseline"
    for key, entry in base.items():
        for name in METHODS:
            assert name in entry, f"{key} missing {name}"
            assert isinstance(entry[name], float)


def test_task_types_and_metrics_are_consistent():
    base = load()
    # AUROC is a percent in [50, 100] for anything on the board; NMAE is a small positive ratio.
    for key, entry in base.items():
        for name in METHODS:
            v = entry[name]
            if entry["type"] == "binary":
                assert entry["metric"] == "auroc"
                assert 50.0 <= v <= 100.0, f"{key}/{name}={v} not a plausible AUROC percent"
            else:
                assert entry["metric"] == "nmae"
                assert 0.0 < v < 10.0, f"{key}/{name}={v} not a plausible NMAE"


def test_regression_and_binary_task_split_matches_the_spec():
    base = load()
    regression = {k for k, e in base.items() if e["type"] == "regression"}
    assert regression == {
        "rel-f1/driver-position", "rel-trial/study-adverse",
        "rel-trial/site-success", "rel-event/user-attendance",
    }


def test_to_display_binary_scales_to_percent():
    assert to_display(0.787, "binary") == pytest.approx(78.7)


def test_to_display_regression_normalizes_mae_by_train_std():
    # raw MAE 4.775 with train-std 10.0 -> NMAE 0.4775 (rel-f1/driver-position's RT value)
    assert to_display(4.775, "regression", train_std=10.0) == pytest.approx(0.4775)
    # already-normalized input passes through
    assert to_display(0.4775, "regression") == pytest.approx(0.4775)


def test_to_display_rejects_nonpositive_std():
    with pytest.raises(ValueError):
        to_display(1.0, "regression", train_std=0.0)


def test_beats_direction_is_metric_aware():
    assert beats(80.0, 78.7, "binary") is True          # higher AUROC wins
    assert beats(70.0, 78.7, "binary") is False
    assert beats(0.40, 0.4775, "regression") is True    # lower NMAE wins
    assert beats(0.50, 0.4775, "regression") is False
    assert beats(1.0, None, "binary") is None           # unreported cell


def test_load_survives_a_method_missing_a_cell(tmp_path):
    doc = {
        "classification": {"rel-f1 driver-dnf": {"RT_from_scratch": "78.7", "GelGT": "-"}},
        "regression": {},
    }
    p = tmp_path / "lb.json"
    p.write_text(json.dumps(doc))
    entry = load(p)["rel-f1/driver-dnf"]
    assert entry["RT (from scratch)"] == pytest.approx(78.7)
    assert "GelGT" not in entry


def test_summary_line_mentions_both_methods():
    entry = load()["rel-f1/driver-dnf"]
    line = summary_line("rel-f1/driver-dnf", 99.0, entry)
    assert "RT (from scratch)" in line and "GelGT" in line and "BEAT" in line
