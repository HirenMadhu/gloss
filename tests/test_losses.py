"""Masked task losses + regression metrics (hermetic)."""
from __future__ import annotations

import pytest
import torch

from gloss.eval.metrics import regression_metrics
from gloss.train.losses import masked_bce, masked_mse, task_loss


def test_masked_mse_zero_when_no_labels():
    pred, y = torch.randn(5), torch.randn(5)
    has = torch.zeros(5, dtype=torch.bool)
    assert float(masked_mse(pred, y, has)) == 0.0


def test_masked_mse_value():
    pred = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 0.0, 3.0])
    has = torch.tensor([True, True, True])
    # squared errors [0, 4, 0] -> mean 4/3
    assert torch.isclose(masked_mse(pred, y, has), torch.tensor(4.0 / 3.0), atol=1e-6)


def test_task_loss_dispatch():
    pred, has = torch.randn(4), torch.ones(4, dtype=torch.bool)
    y = torch.randn(4)
    assert torch.allclose(task_loss(pred, y, has, "regression"), masked_mse(pred, y, has))
    yb = (torch.rand(4) > 0.5).float()
    assert torch.allclose(task_loss(pred, yb, has, "binary"), masked_bce(pred, yb, has))


def test_regression_metrics_known():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(pred, pred)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and abs(m["r2"] - 1.0) < 1e-9
    m2 = regression_metrics(torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0]))
    assert m2["mae"] == 1.0 and abs(m2["rmse"] - 1.0) < 1e-9


# --- regression objective selection (results/WHY_NOT_RT.md §4) -------------------------------------

def test_l1_loss_is_the_mean_absolute_error_over_labelled_seeds():
    """The whole point is that training now optimises the metric RelBench scores."""
    from gloss.train.losses import masked_l1

    pred = torch.tensor([1.0, 5.0, -2.0])
    target = torch.tensor([1.5, 2.0, -2.0])
    mask = torch.tensor([True, True, False])
    assert torch.allclose(masked_l1(pred, target, mask), torch.tensor((0.5 + 3.0) / 2))


def test_task_loss_dispatches_the_regression_objective():
    from gloss.train.losses import masked_huber, masked_l1, masked_mse, task_loss

    pred = torch.tensor([0.0, 4.0])
    target = torch.tensor([1.0, 1.0])
    mask = torch.tensor([True, True])
    for name, fn in (("mse", masked_mse), ("l1", masked_l1), ("huber", masked_huber)):
        assert torch.allclose(task_loss(pred, target, mask, "regression", regression_loss=name),
                              fn(pred, target, mask)), name
    # mse stays the default: every result before 2026-07-31 used it, so a silent switch would make
    # old and new records incomparable while looking identical
    assert torch.allclose(task_loss(pred, target, mask, "regression"),
                          masked_mse(pred, target, mask))


def test_regression_objective_is_ignored_for_binary_not_an_error():
    """Runners pass it unconditionally, so a binary task must not blow up on it."""
    from gloss.train.losses import masked_bce, task_loss

    logits = torch.tensor([0.3, -1.2])
    target = torch.tensor([1.0, 0.0])
    mask = torch.tensor([True, True])
    assert torch.allclose(task_loss(logits, target, mask, "binary", regression_loss="l1"),
                          masked_bce(logits, target, mask))


def test_unknown_regression_objective_raises():
    from gloss.train.losses import task_loss

    with pytest.raises(ValueError, match="unknown regression_loss"):
        task_loss(torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool),
                  "regression", regression_loss="mae")   # the plausible-but-wrong name


def test_l1_beats_mse_at_recovering_a_skewed_median():
    """Why this change exists: on a skewed target, the MSE optimum (mean) is far from the MAE
    optimum (median). study-adverse has skew 39.1, so this is not a hypothetical."""
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(900), torch.full((100,), 50.0)])   # heavy right tail; median 0, mean 5
    mask = torch.ones_like(y, dtype=torch.bool)
    from gloss.train.losses import masked_l1, masked_mse

    def fit(loss_fn):
        c = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([c], lr=0.05)
        for _ in range(400):
            opt.zero_grad(); loss_fn(c.expand_as(y), y, mask).backward(); opt.step()
        return float(c.detach())

    c_mse, c_l1 = fit(masked_mse), fit(masked_l1)
    assert abs(c_l1 - y.median()) < abs(c_mse - y.median())
    # and the L1 fit wins on the MAE metric, which is what RelBench scores
    assert (y - c_l1).abs().mean() < (y - c_mse).abs().mean()


# --- pairwise AUC surrogate for the binary tasks ---------------------------------------------------

def test_pairwise_auc_is_zero_when_every_pair_is_ranked_past_the_margin():
    """Perfectly separated scores with a gap > margin => no pair violates the hinge."""
    from gloss.train.losses import masked_pairwise_auc

    logits = torch.tensor([5.0, 4.0, -4.0, -5.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)
    assert torch.allclose(masked_pairwise_auc(logits, target, mask), torch.zeros(()))


def test_pairwise_auc_penalises_inverted_pairs_and_matches_the_closed_form():
    from gloss.train.losses import masked_pairwise_auc

    logits = torch.tensor([0.0, 2.0])          # one pos (0.0), one neg (2.0) -> inverted
    target = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, dtype=torch.bool)
    # relu(1 - (0 - 2))^2 = 9
    assert torch.allclose(masked_pairwise_auc(logits, target, mask), torch.tensor(9.0))


def test_pairwise_auc_is_zero_and_differentiable_on_a_single_class_batch():
    """A batch with no positives (or no negatives) carries no ranking signal. It must not NaN, and
    it must still return a graph-connected zero so grad shapes stay consistent."""
    from gloss.train.losses import masked_pairwise_auc

    logits = torch.tensor([0.5, 1.5], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    loss = masked_pairwise_auc(logits, target, torch.ones(2, dtype=torch.bool))
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None and torch.all(logits.grad == 0)


def test_pairwise_auc_ranks_better_than_bce_under_imbalance():
    """The reason the surrogate exists: it optimises ranking directly.

    Fit a 1-D scorer on an imbalanced set and check the AUC objective reaches at least as good a
    ranking as BCE. Uses sklearn's AUROC so the assertion is on the real metric, not the surrogate.
    """
    from sklearn.metrics import roc_auc_score

    from gloss.train.losses import masked_bce, masked_pairwise_auc

    torch.manual_seed(0)
    n_pos, n_neg = 20, 400                      # 20:1, harsher than any task here
    x = torch.cat([torch.randn(n_pos) + 1.0, torch.randn(n_neg)]).unsqueeze(1)
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
    mask = torch.ones_like(y, dtype=torch.bool)

    def fit(loss_fn):
        torch.manual_seed(0)
        lin = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss_fn(lin(x).squeeze(-1), y, mask).backward()
            opt.step()
        return roc_auc_score(y.numpy(), lin(x).squeeze(-1).detach().numpy())

    assert fit(masked_pairwise_auc) >= fit(masked_bce) - 1e-6


def test_task_loss_dispatches_the_binary_objective():
    from gloss.train.losses import masked_bce, masked_pairwise_auc, task_loss

    logits = torch.tensor([0.3, -1.2])
    target = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, dtype=torch.bool)
    assert torch.allclose(task_loss(logits, target, mask, "binary", binary_loss="auc"),
                          masked_pairwise_auc(logits, target, mask))
    # bce stays the default so pre-2026-07-31 records remain comparable
    assert torch.allclose(task_loss(logits, target, mask, "binary"), masked_bce(logits, target, mask))
    # and the regression knob is ignored here rather than rejected
    assert torch.allclose(task_loss(logits, target, mask, "binary", regression_loss="l1"),
                          masked_bce(logits, target, mask))


def test_unknown_binary_objective_raises():
    from gloss.train.losses import task_loss

    with pytest.raises(ValueError, match="unknown binary_loss"):
        task_loss(torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool),
                  "binary", binary_loss="aucm")
