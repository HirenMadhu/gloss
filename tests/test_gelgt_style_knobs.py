"""GelGT-style training knobs: percentile clamp, optimizer choice, target scaling, saved predictions.

Ported from GelGT's reference `main_node_ddp.py` after reading it (see `results/WHY_NOT_RT.md`):
it trains regression with L1 on the RAW target, clips predictions to the train target's [2, 98]
percentile before scoring, and uses Adam(wd=1e-5) rather than AdamW(wd=0.01).

The clamp is the load-bearing one. It is an EVAL-time transform, so saving raw predictions lets its
effect be measured on a finished run instead of costing a retrain — these tests pin that property.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gloss.train.losses import task_loss  # noqa: F401  (import guard: module must stay importable)


class _FakeTable:
    def __init__(self, df):
        self.df = df


class _FakeTask:
    """Minimal stand-in for a RelBench regression task (target percentiles are all we need)."""

    target_col = "y"

    def __init__(self, y):
        import pandas as pd

        self._df = pd.DataFrame({"y": y})

    def get_table(self, split, mask_input_cols=True):
        return _FakeTable(self._df)


def test_target_clamp_returns_train_percentiles_in_raw_units():
    from gloss.train.finetune import target_clamp

    y = np.arange(101, dtype="float64")          # 0..100, so percentiles are the values themselves
    lo, hi = target_clamp(_FakeTask(y), 2.0, 98.0)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(98.0)


def test_target_clamp_ignores_nans_and_survives_a_heavy_tail():
    """study-adverse has skew 39.1 — the clamp must be set by the bulk, not dragged by the tail."""
    from gloss.train.finetune import target_clamp

    y = np.concatenate([np.zeros(980), np.full(20, 1e6), [np.nan] * 5])
    lo, hi = target_clamp(_FakeTask(y), 2.0, 98.0)
    assert lo == 0.0
    assert hi < 1e6, "the 98th percentile must exclude the extreme tail, else clamping is a no-op"


def test_clamping_cannot_worsen_mae_when_labels_lie_inside_the_bounds():
    """Why the clamp helps MAE: projecting onto an interval containing the label is non-expansive.

    |clip(p, lo, hi) - y| <= |p - y| for every y in [lo, hi]. So on a task whose labels sit inside
    the train percentiles, the clamp is a free win — it can only pull runaway predictions closer.
    """
    rng = np.random.default_rng(0)
    lo, hi = 0.0, 10.0
    y = rng.uniform(lo, hi, size=500)
    pred = y + rng.normal(0, 20, size=500)        # wild predictions, the failure mode in question
    clamped = np.clip(pred, lo, hi)
    assert np.all(np.abs(clamped - y) <= np.abs(pred - y) + 1e-12)
    assert np.abs(clamped - y).mean() < np.abs(pred - y).mean()


def test_optimizer_switch_builds_adam_or_adamw_and_rejects_others():
    from gloss.train.loop import MoRELitModule

    m = MoRELitModule.__new__(MoRELitModule)      # bypass the heavy graph/model construction
    torch.nn.Module.__init__(m)
    m._p = torch.nn.Parameter(torch.zeros(1))
    m.lr, m.weight_decay = 1e-3, 0.01

    m.optimizer_name = "adamw"
    assert isinstance(m.configure_optimizers(), torch.optim.AdamW)
    m.optimizer_name = "adam"
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam) and not isinstance(opt, torch.optim.AdamW)
    m.optimizer_name = "sgd"
    with pytest.raises(ValueError, match="unknown optimizer"):
        m.configure_optimizers()


def test_raw_target_scaling_disables_standardization():
    """`raw` must make de-standardization the identity, not merely change the loss scale."""
    import inspect

    from gloss.train import finetune

    src = inspect.getsource(finetune.train_prebuilt)
    assert 'target_scaling == "raw"' in src and "mean, std = 0.0, 1.0" in src
    assert "unknown target_scaling" in src


def test_score_pred_and_return_pred_exist_for_post_hoc_evaluation():
    """The whole point of saving predictions: score an eval-time transform without retraining."""
    import inspect

    from gloss.eval import test_eval

    assert callable(test_eval.score_pred)
    sig = inspect.signature(test_eval.evaluate_split).parameters
    assert "return_pred" in sig and "clamp" in sig
    assert "clamp" in inspect.signature(test_eval.predict_split).parameters


def test_predict_split_applies_the_modules_clamp_by_default():
    """A run selected on clamped val metrics must be scored the same way on test unless overridden."""
    import inspect

    from gloss.eval import test_eval

    src = inspect.getsource(test_eval.predict_split)
    assert 'getattr(module, "clamp", None) if clamp is None else clamp' in src
    assert "np.clip(pred, bounds[0], bounds[1])" in src
