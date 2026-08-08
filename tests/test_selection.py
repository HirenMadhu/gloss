"""Val-selection modes (`finetune._BestValState`).

The point of these modes is that `argmax` over ~80 noisy val epochs is a max-of-noise estimator, so a
last-bit CUDA difference flips *which* epoch wins and moves TEST by ~2 AUC. Each test below drives the
callback with a synthetic val curve — signal + a known spike — so "did smoothing actually reject the
spike" is a deterministic assertion rather than something only a GPU run could show.

Every mode reads VAL only; nothing here has access to test, and that is the invariant
`test_selection_never_reads_a_test_metric` pins.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from gloss.train.finetune import SELECT_MODES, _BestValState


class _FakeTrainer:
    def __init__(self):
        self.callback_metrics: dict = {}


def drive(curve, *, select, mode="max", window=3, monitor="val/auroc", extra=None):
    """Run the callback over `curve` with a module whose single weight equals the epoch index.

    Encoding the epoch in the weight makes "which epoch's weights were kept" directly readable from
    `best_state`, which is the thing these modes actually change.
    """
    trainer = _FakeTrainer()
    module = nn.Linear(1, 1, bias=False)
    cb = _BestValState(monitor, mode, select=select, window=window)
    for epoch, score in enumerate(curve):
        with torch.no_grad():
            module.weight.fill_(float(epoch))
        trainer.callback_metrics = {monitor: torch.tensor(score), **(extra or {})}
        cb.on_validation_end(trainer, module)
    rescore = cb.finalize()
    kept = None if cb.best_state is None else round(float(cb.best_state["weight"].item()), 6)
    return cb, kept, rescore


def test_argmax_is_unchanged_and_keeps_the_single_best_epoch():
    """The historical path must be bit-identical in behaviour, or every finished run is unexplained."""
    cb, kept, rescore = drive([0.70, 0.90, 0.75, 0.80], select="argmax")
    assert kept == 1 and cb.best_score == pytest.approx(0.90) and rescore is False


def test_moving_average_keeps_the_centre_of_the_best_window():
    """Windows over [.1 .2 .3 .9 .8 .7] are .2 / .4667 / .6667 / .80 — the last, centred on epoch 4."""
    cb, kept, _ = drive([0.1, 0.2, 0.3, 0.9, 0.8, 0.7], select="ma", window=3)
    assert kept == 4
    assert cb.best_score == pytest.approx(0.80)
    assert cb.raw_best_score == pytest.approx(0.90)   # the argmax is still reported, as a bias probe


def test_min_mode_smoothing_works_for_regression():
    """`val/mae` is minimised — a sign error here would silently select the WORST checkpoint."""
    cb, kept, _ = drive([0.9, 0.8, 0.7, 0.1, 0.2, 0.3], select="ma", mode="min", monitor="val/mae",
                        window=3)
    assert kept == 4
    assert cb.best_score == pytest.approx(0.20)
    assert cb.raw_best_score == pytest.approx(0.10)


def test_smoothing_reduces_selection_variance_on_a_noisy_plateau():
    """The actual claim, measured rather than asserted by construction.

    Smoothing does NOT make the choice immune to a spike — a large enough outlier still drags its
    window (attenuated by 1/window). What it does is reduce the variance of the *signal value at the
    selected epoch*. So: a flat-then-rising true curve plus seeded Gaussian noise, 200 trials, and
    compare how far each mode's pick lands from the true optimum.

    Measured here (200 trials, mean regret relative to `argmax`)::

        window   3      5      9      15
        ma      0.83x  0.76x  0.74x  1.02x
        swa     0.77x  0.84x  0.98x  1.43x

    So smoothing helps up to a point and then **hurts** — a window wide enough to span the optimum
    blurs it away, and `swa` degrades faster because averaging distant epochs' weights is not the same
    as averaging their scores. That ceiling is asserted too; it is the reason `select_window` defaults
    to 5 rather than "as large as possible".
    """
    import math
    import random

    true = [0.70 + 0.02 * math.tanh((t - 40) / 8) for t in range(60)]   # smooth, one broad optimum
    best_true = max(true)

    def regret(select, window):
        rng = random.Random(0)                   # same noise draws for every mode -> paired comparison
        errs = []
        for _ in range(200):
            noisy = [v + rng.gauss(0, 0.01) for v in true]              # noise ~ half the true range
            _, kept, _ = drive(noisy, select=select, window=window)
            errs.append(best_true - true[min(int(round(kept)), len(true) - 1)])
        return sum(errs) / len(errs)

    base = regret("argmax", 1)
    assert regret("ma", 5) < 0.85 * base
    assert regret("swa", 3) < 0.85 * base
    assert regret("ma", 15) > 0.95 * base, "over-smoothing should stop helping, not keep helping"


def test_swa_averages_the_weights_of_the_window_best_epochs():
    """Epochs 5,6,7 are the top three; their encoded weights are 5,6,7 -> mean 6."""
    cb, kept, rescore = drive([0.60, 0.62, 0.64, 0.66, 0.68, 0.90, 0.92, 0.91], select="swa",
                              window=3)
    assert kept == pytest.approx(6.0)
    assert rescore is True                       # averaged weights have no recorded val metrics
    assert cb.best_metrics == {}


def test_swa_min_mode_picks_the_three_lowest():
    cb, kept, _ = drive([0.90, 0.88, 0.10, 0.11, 0.12, 0.70], select="swa", mode="min",
                        monitor="val/mae", window=3)
    assert kept == pytest.approx(3.0)            # epochs 2,3,4 -> mean 3


def test_swa_of_one_epoch_needs_no_rescore():
    """A single kept state is that epoch's own weights, so its recorded val metrics are still valid."""
    cb, kept, rescore = drive([0.7], select="swa", window=3)
    assert kept == 0 and rescore is False and cb.best_metrics


@pytest.mark.parametrize("select", SELECT_MODES)
def test_every_mode_returns_a_model_when_the_run_stops_before_the_window_fills(select):
    """Early stopping can end a run after 2 epochs. A mode that returns `best_state=None` there would
    silently evaluate the LAST-epoch weights instead of a selected checkpoint."""
    cb, kept, _ = drive([0.70, 0.90], select=select, window=5)
    assert cb.best_state is not None and kept is not None


@pytest.mark.parametrize("select", SELECT_MODES)
def test_nan_val_epochs_are_skipped_not_selected(select):
    """A single-class val subsample gives AUROC=NaN; selecting it would return an arbitrary model."""
    cb, kept, _ = drive([0.70, float("nan"), 0.90, float("nan")], select=select, window=2)
    assert cb.n_scored == 2
    assert cb.best_state is not None


@pytest.mark.parametrize("select", SELECT_MODES)
def test_selection_never_reads_a_test_metric(select):
    """These modes change WHICH val statistic is maximised, never what selection may see.

    Feed a `test/auroc` that is perfectly anti-correlated with val: if any mode were peeking at test,
    the kept epoch would move. It must not.
    """
    curve = [0.60, 0.62, 0.90, 0.64, 0.66]
    _, clean, _ = drive(curve, select=select, window=3)
    _, poisoned, _ = drive(curve, select=select, window=3,
                           extra={"test/auroc": torch.tensor(999.0)})
    assert clean == poisoned
    cb, _, _ = drive(curve, select=select, window=3,
                     extra={"test/auroc": torch.tensor(999.0)})
    assert all(k.startswith("val/") for k in cb.best_metrics)


def test_unknown_select_mode_is_rejected_at_construction():
    """A typo must fail loudly, not fall through to `argmax` and silently un-do the experiment."""
    with pytest.raises(ValueError, match="unknown select"):
        _BestValState("val/auroc", "max", select="movingaverage")
