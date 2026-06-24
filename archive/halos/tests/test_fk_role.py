"""Phase 3 — FK-role disambiguation: two FKs into one table get distinct geometry and the model's
attention responds to which role links a pair (the limitation RT explicitly cannot handle).

Hermetic, on the synthetic dual-FK fixture (rel-f1 has no dual-FK-into-one-table)."""
from __future__ import annotations

import dataclasses

import torch

from gloss.data.collate import to_gloss_batch
from gloss.model.attention import RelationalAttention
from gloss.model.bias_generator import BiasGenerator

D_MODEL, H = 16, 4


def _gb(dualfk_batch, dualfk_bundle):
    return to_gloss_batch(dualfk_batch, dualfk_bundle, "user", max_nodes=64)


def test_two_fks_get_distinct_compiled_geometry(dualfk_bundle):
    b = dualfk_bundle
    buyer_mp = b.relation_metapath("f2p_buyer")
    seller_mp = b.relation_metapath("f2p_seller")
    assert buyer_mp != seller_mp
    torch.manual_seed(0)
    bg = BiasGenerator(b.num_metapaths, H, d_text=8)
    geom = bg.compile()
    # the buyer and seller metapaths compile to different Gaussian kernels
    assert not torch.allclose(geom.a[buyer_mp], geom.a[seller_mp])
    assert not torch.allclose(geom.mu[buyer_mp], geom.mu[seller_mp])


def test_role_swap_changes_attention(dualfk_batch, dualfk_bundle):
    b = dualfk_bundle
    gb = _gb(dualfk_batch, dualfk_bundle)
    buyer_mp = b.relation_metapath("f2p_buyer")
    seller_mp = b.relation_metapath("f2p_seller")

    torch.manual_seed(0)
    bg = BiasGenerator(b.num_metapaths, H, d_text=8)
    geom = bg.compile()
    attn = RelationalAttention(D_MODEL, H)
    attn.eval()
    h = torch.randn(gb.num_seeds, gb.n_max, D_MODEL)

    with torch.no_grad():
        out0 = attn(h, gb, geom)

    # swap which pairs are linked by the buyer vs seller role
    mp = gb.metapath_id.clone()
    swapped = mp.clone()
    swapped[mp == buyer_mp] = seller_mp
    swapped[mp == seller_mp] = buyer_mp
    gb_swapped = dataclasses.replace(gb, metapath_id=swapped)

    with torch.no_grad():
        out1 = attn(h, gb_swapped, geom)

    real = gb.pad_mask
    assert not torch.allclose(out0[real], out1[real]), "predictions must respond to the FK role"
