"""The Mixture-of-Experts FFN: top-k gating, dense combine, orthogonality loss (hermetic)."""
from __future__ import annotations

import torch

from gloss.model.moe import MoEFFN


def test_topk_gate_is_sparse_and_normalized():
    torch.manual_seed(0)
    moe = MoEFFN(d_model=16, d_ff=32, d_route=8, num_experts=4, k=2)
    g = moe.gates(torch.randn(3, 5, 8))
    assert g.shape == (3, 5, 4)
    nz = (g > 0).sum(-1)
    assert int(nz.max()) <= 2 and int(nz.min()) >= 1     # at most k experts fire
    assert torch.allclose(g.sum(-1), torch.ones(3, 5), atol=1e-5)


def test_dense_combine_matches_manual_sum():
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, 8, num_experts=4, k=2)
    x = torch.randn(2, 4, 16)
    rf = torch.randn(2, 4, 8)
    y, g = moe(x, rf)
    manual = sum(g[..., e:e + 1] * moe.experts[e](x) for e in range(moe.num_experts))
    assert y.shape == x.shape
    assert torch.allclose(y, manual, atol=1e-5)


def test_ortho_loss_finite_and_positive():
    moe = MoEFFN(16, 32, 8, num_experts=4, k=2)
    loss = moe.ortho_loss()
    assert torch.isfinite(loss) and loss.item() > 0.0


def test_gates_respond_to_route_feat():
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, 8, num_experts=4, k=2)
    g1 = moe.gates(torch.randn(2, 4, 8))
    g2 = moe.gates(torch.randn(2, 4, 8))
    assert not torch.allclose(g1, g2)


# --- ablation additions: S (shared) / C (cosine) / P (top-p) ---

