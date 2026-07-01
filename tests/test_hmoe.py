"""The hierarchical MoE FFN (H): two-level gating, dense combine, level-1 balance (hermetic)."""
from __future__ import annotations

import torch

from gloss.model.moe import HMoEFFN


def test_hmoe_forward_shape_and_level1_gate():
    torch.manual_seed(0)
    h = HMoEFFN(16, 32, 8, n_groups=4, experts_per_group=2, k2=1)
    x = torch.randn(2, 5, 16)
    rf = torch.randn(2, 5, 8)
    y, p1 = h(x, rf)
    assert y.shape == x.shape
    assert p1.shape == (2, 5, 4)                             # soft over groups
    assert torch.allclose(p1.sum(-1), torch.ones(2, 5), atol=1e-5)


def test_hmoe_ortho_loss_finite_and_nonneg():
    torch.manual_seed(0)
    h = HMoEFFN(16, 32, 8, n_groups=4, experts_per_group=2, k2=1)
    h(torch.randn(2, 5, 16), torch.randn(2, 5, 8))          # sets the level-1 balance term
    loss = h.ortho_loss()
    assert torch.isfinite(loss) and loss.item() > 0.0        # gate-row ortho (>0) + occupancy penalty (>=0)


def test_hmoe_grads_reach_level1_gate_and_experts():
    """k2=1 (default) is a hard within-group top-1, so level-1 gate + experts train (level-2 does not)."""
    torch.manual_seed(0)
    h = HMoEFFN(16, 32, 8, n_groups=4, experts_per_group=2, k2=1)
    y, _ = h(torch.randn(4, 8, 16), torch.randn(4, 8, 8))
    (y.sum() + h.ortho_loss()).backward()
    assert h.g1.weight.grad is not None and h.g1.weight.grad.abs().sum() > 0
    assert h.experts[0][0].w1.weight.grad is not None and h.experts[0][0].w1.weight.grad.abs().sum() > 0


def test_hmoe_level2_gate_trains_when_k2_ge_2():
    """With k2>=2 the within-group softmax is non-degenerate, so the level-2 gate receives gradient."""
    torch.manual_seed(0)
    h = HMoEFFN(16, 32, 8, n_groups=4, experts_per_group=3, k2=2)
    y, _ = h(torch.randn(4, 8, 16), torch.randn(4, 8, 8))
    (y.sum() + h.ortho_loss()).backward()
    assert h.g2[0].weight.grad is not None and h.g2[0].weight.grad.abs().sum() > 0
