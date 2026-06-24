"""Phase 3 — relational attention: shapes, padding, masking, and temporal-bias gating."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from gloss.data.collate import to_gloss_batch
from gloss.model.attention import RelationalAttention
from gloss.model.bias_generator import BiasGenerator

D_MODEL, H = 16, 4


def _setup(dualfk_batch, dualfk_bundle):
    gb = to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)
    torch.manual_seed(0)
    bg = BiasGenerator(dualfk_bundle.num_metapaths, H, d_text=8)
    geom = bg.compile()
    attn = RelationalAttention(D_MODEL, H).eval()
    h = torch.randn(gb.num_seeds, gb.n_max, D_MODEL)
    return gb, geom, attn, h


def test_attention_shape_finite_padzero(dualfk_batch, dualfk_bundle):
    gb, geom, attn, h = _setup(dualfk_batch, dualfk_bundle)
    with torch.no_grad():
        out = attn(h, gb, geom)
    assert out.shape == (gb.num_seeds, gb.n_max, D_MODEL)
    assert torch.isfinite(out).all()           # no NaN even though padded rows are fully masked
    assert torch.count_nonzero(out[~gb.pad_mask]) == 0


def test_temporal_bias_gating_changes_output(dualfk_batch, dualfk_bundle):
    gb, geom, attn, h = _setup(dualfk_batch, dualfk_bundle)
    with torch.no_grad():
        out0 = attn(h, gb, geom)
        # zero out the dimensionless lag on temporal pairs -> Gaussian term shifts -> output changes
        gb2 = dataclasses.replace(gb, tau=gb.tau + 3.0)
        out1 = attn(h, gb2, geom)
    real = gb.pad_mask
    assert not torch.allclose(out0[real], out1[real])


def test_masked_pairs_do_not_leak(dualfk_batch, dualfk_bundle):
    gb, geom, attn, h = _setup(dualfk_batch, dualfk_bundle)
    # the fixture's 3-node subgraph is fully attendable; mask off one off-diagonal pair and confirm
    # changing its (now-masked) bias has no effect on the output.
    am = gb.attend_mask.clone()
    am[:, 0, 1] = False
    am[:, 1, 0] = False
    gb_masked = dataclasses.replace(gb, attend_mask=am)
    with torch.no_grad():
        base = attn(h, gb_masked, geom)
        # perturb metapath id of the masked pair; output must not move
        mp = gb.metapath_id.clone()
        mp[:, 0, 1] = (mp[:, 0, 1] + 1) % geom.num_metapaths
        out = attn(h, dataclasses.replace(gb_masked, metapath_id=mp), geom)
    assert torch.allclose(base[gb.pad_mask], out[gb.pad_mask], atol=1e-6)


@pytest.mark.slow
def test_flex_matches_sdpa(dualfk_batch, dualfk_bundle):
    """FlexAttention backend parity with SDPA on real (non-pad) rows. Marked slow (torch.compile)."""
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention needs a GPU/Triton backend")
    dev = torch.device("cuda")
    gb, geom, _attn, h = _setup(dualfk_batch, dualfk_bundle)
    gb, h = gb.to(dev), h.to(dev)
    geom = dataclasses.replace(geom, a=geom.a.to(dev), mu=geom.mu.to(dev),
                               sigma=geom.sigma.to(dev), b=geom.b.to(dev))
    torch.manual_seed(1)
    sdpa = RelationalAttention(D_MODEL, H, attn_impl="sdpa").to(dev).eval()
    flex = RelationalAttention(D_MODEL, H, attn_impl="flex").to(dev).eval()
    flex.load_state_dict(sdpa.state_dict())   # identical weights
    with torch.no_grad():
        out_sdpa = sdpa(h, gb, geom)
        try:
            out_flex = flex(h, gb, geom)
        except Exception as e:                # flex needs a recent triton/compile stack
            pytest.skip(f"flex_attention unavailable in this env: {e}")
    assert torch.allclose(out_sdpa[gb.pad_mask], out_flex[gb.pad_mask], atol=1e-3)
