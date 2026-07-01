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

def test_shared_expert_is_added_as_residual():
    """S: y = routed sum + shared(x); the base (no-shared) MoE equals just the routed sum."""
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, 8, num_experts=4, k=2, use_shared=True)
    x = torch.randn(2, 4, 16)
    rf = torch.randn(2, 4, 8)
    y, g = moe(x, rf)
    routed = sum(g[..., e:e + 1] * moe.experts[e](x) for e in range(moe.num_experts))
    assert moe.shared is not None
    assert torch.allclose(y, routed + moe.shared(x), atol=1e-5)
    assert not torch.allclose(y, routed, atol=1e-5)          # the shared branch actually contributes


def test_cosine_router_gates_and_ortho_on_keys():
    """C: cosine-normalized logits over learnable keys; top-k still sparse + sum to 1; ortho on keys."""
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, 8, num_experts=4, k=2, cosine=True, tau=0.3)
    assert hasattr(moe, "keys") and not hasattr(moe, "router")
    g = moe.gates(torch.randn(3, 5, 8))
    nz = (g > 0).sum(-1)
    assert int(nz.max()) <= 2 and int(nz.min()) >= 1
    assert torch.allclose(g.sum(-1), torch.ones(3, 5), atol=1e-5)
    loss = moe.ortho_loss()
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert moe.keys.grad is not None and moe.keys.grad.abs().sum() > 0   # decorrelates the keys


def test_top_p_support_is_adaptive_and_normalized():
    """P: keep the smallest set reaching cumulative mass P; support varies but >=1, rows still sum to 1."""
    torch.manual_seed(0)
    moe = MoEFFN(16, 32, 8, num_experts=8, k=2, top_p=0.5)
    g = moe.gates(torch.randn(50, 8))
    nz = (g > 0).sum(-1)
    assert int(nz.min()) >= 1 and int(nz.max()) <= 8
    assert torch.allclose(g.sum(-1), torch.ones(50), atol=1e-5)


def test_top_p_larger_mass_keeps_at_least_as_many_experts():
    """Higher P never keeps fewer experts than lower P (same router weights)."""
    torch.manual_seed(0)
    rf = torch.randn(200, 8)
    lo = MoEFFN(16, 32, 8, num_experts=8, top_p=0.2)
    hi = MoEFFN(16, 32, 8, num_experts=8, top_p=0.9)
    hi.load_state_dict(lo.state_dict())                      # isolate the effect of top_p
    k_lo = (lo.gates(rf) > 0).sum(-1).float().mean()
    k_hi = (hi.gates(rf) > 0).sum(-1).float().mean()
    assert float(k_hi) >= float(k_lo)
