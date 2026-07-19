"""SetJoin model contracts.

Hermetic tests exercise the sub-modules on raw tensors (no CellEncoder — the repo's synthetic fixtures
carry no pytorch-frame stats); full-model tests run on the cached rel-f1 (repo convention).
"""
from __future__ import annotations

import torch

from gloss.setjoin.model import (AxialCellEncoder, ElemSignature, MoELayer, RowPool, SetEncoder,
                                 SetJoin, WideEncoder)

from .conftest import rel_f1_available

D = 16


# ---------- hermetic sub-module contracts ----------

def test_row_pool_shapes_and_grads():
    pool = RowPool(D)
    cells = torch.randn(5, 7, D, requires_grad=True)
    out = pool(cells)
    assert out.shape == (5, D)
    out.sum().backward()
    assert cells.grad is not None and pool.w1.weight.grad is not None


def test_wide_encoder_cls_and_all_pad_row():
    enc = WideEncoder(D, n_layers=1, n_heads=2, dropout=0.0).eval()
    h = torch.randn(2, 6, D)
    is_pad = torch.zeros(2, 6, dtype=torch.bool)
    is_pad[1] = True                                       # degenerate: all-pad wide row
    out = enc(h * ~is_pad.unsqueeze(-1), is_pad)
    assert out.shape == (2, D) and torch.isfinite(out).all()


def test_set_encoder_permutation_invariance():
    enc = SetEncoder(D, n_layers=2, n_heads=2, n_pma=3, dropout=0.0).eval()
    E = torch.randn(2, 5, D)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    seed = torch.randn(2, D)
    out = enc(E, mask, seed)
    perm = torch.tensor([2, 0, 1, 4, 3])                   # permutes real AND pad slots
    out_p = enc(E[:, perm], mask[:, perm], seed)
    assert torch.allclose(out, out_p, atol=1e-5)


def test_moe_layer_routes_and_flows():
    d_sig = 8
    lyr = MoELayer(D, n_heads=2, d_ff=2 * D, d_route=d_sig, num_experts=3, k=2,
                   dropout=0.0, route_on="signature")
    x = torch.randn(2, 5, D, requires_grad=True)
    z = torch.randn(2, 5, d_sig)
    kpm = torch.zeros(2, 5, dtype=torch.bool)
    out = lyr(x, z, kpm)
    assert out.shape == (2, 5, D) and torch.isfinite(out).all()
    # different signatures -> different routing -> different output
    out2 = lyr(x, torch.randn(2, 5, d_sig), kpm)
    assert not torch.allclose(out, out2, atol=1e-6)
    out.sum().backward()
    assert lyr.ffn.router.weight.grad is not None and x.grad is not None
    assert float(lyr.ffn.ortho_loss()) >= 0.0
    # set-side permutation invariance survives the MoE layer (z permutes with x)
    lyr.eval()
    perm = torch.tensor([3, 1, 4, 0, 2])
    a = lyr(x.detach(), z, kpm)
    b = lyr(x.detach()[:, perm], z[:, perm], kpm[:, perm])
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_elem_signature_distinguishes_hops():
    """Hop-2 elements must route differently from hop-1: the row-level signature reads elem_hop,
    so flipping an element's hop tag (all else equal) must change its z."""
    from types import SimpleNamespace

    sig = ElemSignature(num_node_types=3, num_fk_roles=4, d_sig=16).eval()

    def jb(hop):
        return SimpleNamespace(
            elem_table_idxs=torch.tensor([[1, 2]]),
            elem_rel_idxs=torch.tensor([[1, 3]]),
            elem_row_time=torch.tensor([[10.0, 20.0]], dtype=torch.float64),
            elem_is_timed=torch.ones(1, 2, dtype=torch.bool),
            elem_hop=torch.full((1, 2), hop, dtype=torch.long),
            seed_time=torch.tensor([100.0], dtype=torch.float64),
        )

    z1, z2 = sig(jb(1)), sig(jb(2))
    assert z1.shape == z2.shape == (1, 3, 16)                 # null sig prepended
    assert torch.allclose(z1[:, 0], z2[:, 0])                 # null element unaffected
    assert not torch.allclose(z1[:, 1:], z2[:, 1:], atol=1e-6)


def test_moe_layer_shared_expert_arm():
    """use_shared=True (the S addition): the always-on expert exists, contributes to the output,
    and receives gradients; the base arm keeps ffn.shared=None so default behavior is unchanged."""
    d_sig = 8
    base = MoELayer(D, n_heads=2, d_ff=2 * D, d_route=d_sig, num_experts=3, k=2,
                    dropout=0.0, route_on="signature")
    assert base.ffn.shared is None
    lyr = MoELayer(D, n_heads=2, d_ff=2 * D, d_route=d_sig, num_experts=3, k=2,
                   dropout=0.0, route_on="signature", use_shared=True)
    assert lyr.ffn.shared is not None
    x = torch.randn(2, 5, D, requires_grad=True)
    z = torch.randn(2, 5, d_sig)
    kpm = torch.zeros(2, 5, dtype=torch.bool)
    out = lyr(x, z, kpm)
    assert out.shape == (2, 5, D) and torch.isfinite(out).all()
    out.sum().backward()
    assert lyr.ffn.shared.w1.weight.grad is not None
    assert float(lyr.ffn.shared.w1.weight.grad.abs().sum()) > 0
    assert lyr.ffn.router.weight.grad is not None and x.grad is not None


def test_set_encoder_empty_set_and_seed_conditioning():
    enc = SetEncoder(D, n_layers=1, n_heads=2, n_pma=2, dropout=0.0).eval()
    E = torch.zeros(2, 4, D)
    empty = torch.zeros(2, 4, dtype=torch.bool)
    seed = torch.randn(2, D)
    out = enc(E, empty, seed)
    assert torch.isfinite(out).all()
    # grads reach the learned null element even with an empty set
    enc.train()
    out2 = enc(E, empty, seed)
    out2.sum().backward()
    assert enc.null_elem.grad is not None and float(enc.null_elem.grad.abs().sum()) > 0
    # the readout is conditioned on the seed representation. NOTE: this needs >= 2 attention keys —
    # over a single key (empty set -> null element only) softmax is 1.0 whatever the query, so seed
    # conditioning is inert by construction for empty sets (the head still sees seed_repr directly).
    enc.eval()
    E2 = torch.randn(2, 4, D)
    some = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    assert not torch.allclose(enc(E2, some, seed), enc(E2, some, seed + 1.0), atol=1e-5)


def test_axial_encoder_shapes_and_nan_safety():
    # seg 1 has ZERO rows of this type: its all-pad grid row must not leak NaN anywhere
    enc = AxialCellEncoder(D, n_layers=1, n_heads=2, dropout=0.0,
                           route_on="dense", d_ff=2 * D, d_route=D, num_experts=2, k=1).eval()
    cell = torch.randn(5, 3, D)
    seg = torch.tensor([0, 0, 0, 2, 2])
    out = enc(cell, seg, num_seeds=3, z=None)
    assert out.shape == cell.shape and torch.isfinite(out).all()


def test_axial_column_attention_mixes_rows_within_seed_only():
    enc = AxialCellEncoder(D, n_layers=1, n_heads=2, dropout=0.0,
                           route_on="dense", d_ff=2 * D, d_route=D, num_experts=2, k=1).eval()
    cell = torch.randn(4, 3, D)
    seg = torch.tensor([0, 0, 1, 1])
    base = enc(cell, seg, num_seeds=2, z=None)
    pert = cell.clone()
    # NOTE: perturb with a random vector — a constant shift lies in pre-norm LayerNorm's null space
    torch.manual_seed(1)
    pert[0] += torch.randn_like(pert[0])                    # perturb row 0 (seed 0)
    out = enc(pert, seg, num_seeds=2, z=None)
    assert not torch.allclose(out[1], base[1], atol=1e-6)   # same-seed row sees it (column attention)
    assert torch.allclose(out[2:], base[2:], atol=1e-6)     # other seed is isolated (per-seed grids)


def test_axial_row_attention_mixes_columns():
    enc = AxialCellEncoder(D, n_layers=1, n_heads=2, dropout=0.0,
                           route_on="dense", d_ff=2 * D, d_route=D, num_experts=2, k=1).eval()
    cell = torch.randn(1, 4, D)                             # single row, 4 columns -> only row-level acts
    seg = torch.tensor([0])
    base = enc(cell, seg, num_seeds=1, z=None)
    pert = cell.clone()
    torch.manual_seed(1)
    pert[0, 0] += torch.randn_like(pert[0, 0])              # random, not constant (LayerNorm null space)
    out = enc(pert, seg, num_seeds=1, z=None)
    assert not torch.allclose(out[0, 2], base[0, 2], atol=1e-6)   # col 0 change reaches col 2


def test_axial_moe_arm_routes_and_balances():
    d_sig = 8
    enc = AxialCellEncoder(D, n_layers=1, n_heads=2, dropout=0.0,
                           route_on="signature", d_ff=2 * D, d_route=d_sig, num_experts=3, k=2)
    cell = torch.randn(4, 3, D, requires_grad=True)
    seg = torch.tensor([0, 0, 1, 1])
    z = torch.randn(4, 3, d_sig)
    out = enc(cell, seg, num_seeds=2, z=z)
    assert torch.isfinite(out).all()
    (out.sum() + enc.ortho_loss()).backward()
    assert enc.blocks[0].ffn.router.weight.grad is not None
    assert cell.grad is not None and float(enc.ortho_loss()) >= 0.0


# ---------- full model on the cached real dataset ----------

def _real_jb_and_model(encoder="hash", route_on="signature"):
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.setjoin.collate import to_join_batch
    from gloss.setjoin.paths import setjoin_neighbors
    from gloss.train.finetune import name_embeddings

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout=8),
                         batch_size=8)
    jb = to_join_batch(next(iter(loader)), bundle, task.entity_table, wide_len=64, set_size=32)
    name_emb = name_embeddings(bundle, "rel-f1", encoder=encoder, d_text=32)
    # n_wide_layers=2, NOT 1: with a single wide layer the non-CLS tokens' FFN outputs never reach
    # the CLS readout (no attention follows the FFN), so wide-signature routing would get zero grad.
    model = SetJoin(bundle, name_emb, task.entity_table, d_model=32, n_heads=2,
                    n_wide_layers=2, n_set_layers=1, n_pma=2, dropout=0.0,
                    route_on=route_on, num_experts=3, k=2, d_sig=16)
    return jb, model


@rel_f1_available
def test_forward_shapes_grads_and_budget():
    # default arm = signature-routed MoE (the method): aux is the router ortho loss, > 0
    jb, model = _real_jb_and_model()
    logits, aux = model(jb)
    assert logits.shape == (jb.num_seeds, 1) and torch.isfinite(logits).all()
    assert torch.isfinite(aux) and float(aux) > 0.0
    assert sum(p.numel() for p in model.parameters()) < 30e6
    (logits.sum() + aux).backward()
    for name in ("row_pool.w1.weight", "path_emb.weight", "fk_emb.weight", "table_emb.weight",
                 "set_enc.null_elem", "encoder.name_proj.weight",
                 "wide_enc.stack.layers.0.ffn.router.weight",
                 "set_enc.stack.layers.0.ffn.router.weight",
                 "wide_sig.schema_proj.weight", "elem_sig.fk_emb.weight"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and float(g.abs().sum()) > 0, name


@rel_f1_available
def test_dense_arm_has_no_moe_and_zero_aux():
    jb, model = _real_jb_and_model(route_on="dense")
    logits, aux = model(jb)
    assert torch.isfinite(logits).all() and float(aux) == 0.0
    assert model.wide_sig is None and model.elem_sig is None
    assert not any("ffn.router" in n for n, _ in model.named_parameters())


@rel_f1_available
def test_three_level_model_forward_and_grads():
    # v4 arm: axial (row+column) + set attention + signature-routed MoE everywhere
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.setjoin.collate import to_join_batch
    from gloss.setjoin.paths import setjoin_neighbors
    from gloss.train.finetune import name_embeddings

    bundle = build_gloss_graph("rel-f1")
    task = get_task("rel-f1", "driver-dnf", download=False)
    loader = make_loader(bundle, task, "train", num_neighbors=setjoin_neighbors(bundle, fanout=8),
                         batch_size=8)
    jb = to_join_batch(next(iter(loader)), bundle, task.entity_table, wide_len=64, set_size=32)
    name_emb = name_embeddings(bundle, "rel-f1", encoder="hash", d_text=32)
    model = SetJoin(bundle, name_emb, task.entity_table, d_model=32, n_heads=2,
                    n_wide_layers=2, n_set_layers=1, n_pma=2, dropout=0.0,
                    route_on="signature", num_experts=3, k=2, d_sig=16, n_axial_layers=1)
    logits, aux = model(jb)
    assert logits.shape == (jb.num_seeds, 1) and torch.isfinite(logits).all()
    assert torch.isfinite(aux) and float(aux) > 0.0
    (logits.sum() + aux).backward()
    for name in ("axial.blocks.0.attn_r.in_proj_weight", "axial.blocks.0.attn_c.in_proj_weight",
                 "axial.blocks.0.ffn.router.weight"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and float(g.abs().sum()) > 0, name


@rel_f1_available
def test_signatures_are_value_free():
    # the routing signal must not depend on cell VALUES — only on schema ids and times
    jb, model = _real_jb_and_model()
    z_w, z_e = model.wide_sig(jb), model.elem_sig(jb)
    import torch_frame

    for tf in jb.tf_dict.values():                          # perturb every numerical value
        if torch_frame.numerical in tf.feat_dict:
            tf.feat_dict[torch_frame.numerical] += 123.0
    assert torch.equal(z_w, model.wide_sig(jb)) and torch.equal(z_e, model.elem_sig(jb))


@rel_f1_available
def test_end_to_end_set_permutation_invariance():
    from dataclasses import replace

    jb, model = _real_jb_and_model()
    model.eval()
    logits, _ = model(jb)
    N = jb.set_size
    perm = torch.randperm(N)
    inv = torch.argsort(perm)
    jb_p = replace(
        jb,
        elem_mask=jb.elem_mask[:, perm], elem_rel_idxs=jb.elem_rel_idxs[:, perm],
        elem_table_idxs=jb.elem_table_idxs[:, perm], elem_row_time=jb.elem_row_time[:, perm],
        elem_is_timed=jb.elem_is_timed[:, perm], elem_hop=jb.elem_hop[:, perm],
        elem_rows={nt: (b, inv[n], r, p) for nt, (b, n, r, p) in jb.elem_rows.items()},
    )
    logits_p, _ = model(jb_p)
    assert torch.allclose(logits, logits_p, atol=1e-5)


@rel_f1_available
def test_empty_set_and_missing_marker_paths_are_live():
    from dataclasses import replace

    jb, model = _real_jb_and_model()
    model.eval()
    logits, _ = model(jb)
    # empty union set: finite output, and it differs from the populated one
    jb_empty = replace(jb, elem_mask=torch.zeros_like(jb.elem_mask), elem_rows={})
    logits_e, _ = model(jb_empty)
    assert torch.isfinite(logits_e).all()
    assert not torch.allclose(logits, logits_e, atol=1e-5)
    # flipping a real wide cell into a missing marker changes the output (missing_emb is live)
    wm = jb.wide_missing.clone()
    wm[0, 0] = True
    logits_m, _ = model(replace(jb, wide_missing=wm))
    assert not torch.allclose(logits[0], logits_m[0], atol=1e-6)
