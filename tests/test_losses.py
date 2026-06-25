"""Masked task losses + regression metrics (hermetic)."""
from __future__ import annotations

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
