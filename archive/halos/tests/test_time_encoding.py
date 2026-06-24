"""Phase 2 — dimensionless time: tau formula, validity, scale-invariance, Bochner features."""
from __future__ import annotations

import math

import torch

from gloss.model.time_encoding import BochnerTime, node_tau


def test_node_tau_formula_and_validity():
    seed_time = torch.tensor([100.0, 100.0], dtype=torch.float64)
    row_time = torch.tensor([[90.0, 100.0, 0.0], [50.0, 0.0, 0.0]], dtype=torch.float64)
    is_timed = torch.tensor([[True, True, False], [True, False, False]])
    t_ctx = torch.tensor([5.0, 5.0], dtype=torch.float64)
    tau, valid = node_tau(seed_time, row_time, t_ctx, is_timed)
    # node 0,0: gap=10 -> log(10/5)=log2 ; valid
    assert math.isclose(tau[0, 0].item(), math.log(10.0 / 5.0), abs_tol=1e-12)
    assert bool(valid[0, 0])
    # node 0,1: gap=0 -> invalid (tau 0)
    assert not bool(valid[0, 1]) and tau[0, 1].item() == 0.0
    # node 0,2: timeless -> invalid
    assert not bool(valid[0, 2])
    assert bool(valid[1, 0]) and not bool(valid[1, 1])


def test_node_tau_is_scale_invariant():
    g = torch.Generator().manual_seed(0)
    seed_time = torch.tensor([1000.0, 500.0], dtype=torch.float64)
    row_time = torch.tensor([[10.0, 200.0, 0.0], [50.0, 100.0, 0.0]], dtype=torch.float64)
    is_timed = torch.tensor([[True, True, True], [True, True, True]])
    t_ctx = torch.tensor([7.0, 3.0], dtype=torch.float64)
    tau0, _ = node_tau(seed_time, row_time, t_ctx, is_timed)
    for c in (0.01, 3.3, 1000.0):
        tau1, _ = node_tau(seed_time * c, row_time * c, t_ctx * c, is_timed)
        assert torch.allclose(tau0, tau1, atol=1e-9)


def test_bochner_shape_and_determinism():
    bt = BochnerTime(n_freq=16)
    tau = torch.randn(4, 5)
    f1 = bt(tau)
    f2 = bt(tau)
    assert f1.shape == (4, 5, 16)
    assert torch.equal(f1, f2)
    assert torch.isfinite(f1).all()
    assert bt.out_dim == 16
