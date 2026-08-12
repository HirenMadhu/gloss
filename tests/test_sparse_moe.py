"""Sparse MoE dispatch, shared-expert pools, and gradient checkpointing.

The load-bearing claim is that ``dispatch='sparse'`` is a *compute* change and not a *model* change:
the gate was always sparse (``k`` of ``M`` weights non-zero), only the combine evaluated all ``M``
experts and multiplied the rest by zero. So the sparse path must reproduce the dense path's output
**and its gradients**, not merely resemble them — otherwise every measured result in this repo would
silently move the first time a pretraining config turns it on.

Tolerances are fp32 accumulation slack, not modelling slack: the two paths sum the same products in a
different order (``index_add`` over token buckets vs a running sum over experts).
"""
from __future__ import annotations

import pytest
import torch

from gloss.model.moe import MoEFFN, sparse_combine
from gloss.model.row_level import RowMoE

from .test_row_level import D_MODEL, D_SIG, stub_batch

D, DFF, DROUTE = 16, 32, 8


def _paired(cls_kwargs: dict, **ctor):
    """Two modules of the same class with identical weights, differing only in `dispatch`."""
    torch.manual_seed(0)
    dense = ctor.pop("cls")(**cls_kwargs, dispatch="dense", **ctor)
    torch.manual_seed(0)
    sparse = type(dense)(**cls_kwargs, dispatch="sparse", **ctor)
    sparse.load_state_dict(dense.state_dict())
    return dense, sparse


# --------------------------------------------------------------------------------------- cell MoE


@pytest.mark.parametrize("num_shared", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_sparse_matches_dense_forward(k, num_shared):
    dense, sparse = _paired({"d_model": D, "d_ff": DFF, "d_route": DROUTE},
                            cls=MoEFFN, num_experts=4, k=k, num_shared=num_shared)
    x, z = torch.randn(3, 7, D), torch.randn(3, 7, DROUTE)
    yd, gd = dense(x, z)
    ys, gs = sparse(x, z)
    assert torch.allclose(gd, gs, atol=0), "the gate itself must be identical, not merely close"
    assert torch.allclose(yd, ys, atol=1e-6), (yd - ys).abs().max()


@pytest.mark.parametrize("num_shared", [0, 2])
def test_sparse_matches_dense_gradients(num_shared):
    """Gradients too — the -inf entries in the gate carry none, so nothing is lost by not running
    the experts that the top-k excluded."""
    dense, sparse = _paired({"d_model": D, "d_ff": DFF, "d_route": DROUTE},
                            cls=MoEFFN, num_experts=4, k=2, num_shared=num_shared)
    x, z = torch.randn(5, 9, D), torch.randn(5, 9, DROUTE)
    for m in (dense, sparse):
        m(x.clone(), z.clone())[0].square().sum().backward()
    for (n, pd), (_, ps) in zip(dense.named_parameters(), sparse.named_parameters()):
        assert pd.grad is not None and ps.grad is not None, n
        assert torch.allclose(pd.grad, ps.grad, atol=1e-5), (n, (pd.grad - ps.grad).abs().max())


def test_sparse_valid_zeroes_excluded_tokens_and_leaves_the_rest_alone():
    """`valid` is the padding saving. It must zero exactly the excluded rows and change nothing else."""
    moe = MoEFFN(D, DFF, DROUTE, num_experts=4, k=2, num_shared=1, dispatch="sparse")
    x, z = torch.randn(2, 6, D), torch.randn(2, 6, DROUTE)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, 3] = valid[1, 5] = False

    full, _ = moe(x, z)
    part, _ = moe(x, z, valid=valid)
    assert torch.equal(part[0, 3], torch.zeros(D))
    assert torch.equal(part[1, 5], torch.zeros(D))
    assert torch.allclose(part[valid], full[valid], atol=1e-6)


def test_dense_ignores_valid_so_measured_results_cannot_move():
    """The dense combine is the path every reported number came from; `valid` must not touch it."""
    moe = MoEFFN(D, DFF, DROUTE, num_experts=4, k=2, dispatch="dense")
    x, z = torch.randn(2, 6, D), torch.randn(2, 6, DROUTE)
    valid = torch.zeros(2, 6, dtype=torch.bool)
    assert torch.equal(moe(x, z)[0], moe(x, z, valid=valid)[0])


def test_shared_experts_are_ungated_and_additive():
    moe = MoEFFN(D, DFF, DROUTE, num_experts=4, k=2, num_shared=2, dispatch="dense")
    x, z = torch.randn(4, D), torch.randn(4, DROUTE)
    g = moe.gates(z)
    routed = sum(g[..., e:e + 1] * moe.experts[e](x) for e in range(4))
    assert torch.allclose(moe(x, z)[0] - routed, sum(m(x) for m in moe.shared), atol=1e-6)


def test_cell_moe_defaults_to_routed_only():
    """The measured cell-level asymmetry: routed-only here, shared+routed at the row level."""
    assert MoEFFN(D, DFF, DROUTE).shared is None
    assert MoEFFN(D, DFF, DROUTE).dispatch == "dense", "dense must stay the default"


def test_unknown_dispatch_raises():
    with pytest.raises(ValueError, match="dispatch"):
        MoEFFN(D, DFF, DROUTE, dispatch="grouped")


def test_sparse_combine_skips_experts_with_no_tokens():
    """An expert nobody routes to must contribute nothing — and get no gradient, which is the DDP
    hazard documented on `sparse_combine`."""
    experts = torch.nn.ModuleList(torch.nn.Linear(D, D, bias=False) for _ in range(3))
    x = torch.randn(4, D)
    topi = torch.zeros(4, 1, dtype=torch.long)            # everything to expert 0
    gates = torch.zeros(4, 3)
    gates[:, 0] = 1.0
    sparse_combine(x, gates, topi, experts).square().sum().backward()
    assert experts[0].weight.grad is not None
    assert experts[1].weight.grad is None and experts[2].weight.grad is None


# ---------------------------------------------------------------------------------------- row MoE


@pytest.mark.parametrize("num_shared", [0, 1, 3])
def test_row_moe_sparse_matches_dense_on_valid_rows(num_shared):
    """The row level always masked padding out of its aux/diag but never out of its combine, so here
    the two paths agree on valid rows and the sparse one additionally zeroes the padding."""
    cb, _K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    dense, sparse = _paired({"d_model": D_MODEL, "d_ff": 4 * D_MODEL, "d_sig": D_SIG},
                            cls=RowMoE, num_experts=4, k=2, num_shared=num_shared)
    u, z = torch.randn(B, R, D_MODEL), torch.randn(B, R, D_SIG)
    yd, auxd, _ = dense(u, z, cb)
    ys, auxs, _ = sparse(u, z, cb)
    v = cb.row_valid
    assert torch.allclose(yd[v], ys[v], atol=1e-5), (yd[v] - ys[v]).abs().max()
    assert torch.allclose(auxd, auxs, atol=1e-6), "aux reads gates only; dispatch must not touch it"
    if (~v).any():
        assert torch.allclose(ys[~v], u[~v], atol=0), "padding rows keep their residual, add nothing"


def test_row_moe_num_shared_overrides_use_shared():
    assert RowMoE(D_MODEL, 4 * D_MODEL, D_SIG).num_shared == 1              # use_shared default True
    assert RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=False).shared is None
    assert len(RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, num_shared=3).shared) == 3
    assert RowMoE(D_MODEL, 4 * D_MODEL, D_SIG, use_shared=True, num_shared=0).shared is None


# --------------------------------------------------------------- gradient checkpointing (substrate)


def _substrate(**kw):
    from gloss.model.two_level import TwoLevelSubstrate

    torch.manual_seed(0)
    d_text = 12
    return TwoLevelSubstrate(
        D_MODEL, 2 * D_MODEL, D_SIG,
        torch.randn(4, d_text), torch.randn(5, d_text), torch.randn(9, d_text),
        n_blocks=2, n_heads=4, **kw,
    )


def test_grad_checkpointing_changes_neither_loss_nor_gradients():
    cb, _K = stub_batch()
    B, R = cb.num_seeds, cb.adj_role.shape[1]
    S = cb.seq_len
    plain = _substrate()
    ckpt = _substrate(grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    plain.train(), ckpt.train()

    x, z = torch.randn(B, S, D_MODEL), torch.randn(B, S, D_SIG)
    outs = []
    for m in (plain, ckpt):
        h, u, aux, _ = m(x.clone(), cb, z=z.clone())
        (h.square().sum() + u.square().sum() + aux).backward()
        outs.append(h)
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    for (n, a), (_, b) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert a.grad is not None and b.grad is not None, n
        assert torch.allclose(a.grad, b.grad, atol=1e-4), (n, (a.grad - b.grad).abs().max())


def test_mean_aux_divides_by_depth_and_is_off_by_default():
    cb, _K = stub_batch()
    B, R, S = cb.num_seeds, cb.adj_role.shape[1], cb.seq_len
    x, z = torch.randn(B, S, D_MODEL), torch.randn(B, S, D_SIG)
    summed = _substrate()
    meaned = _substrate(mean_aux=True)
    meaned.load_state_dict(summed.state_dict())
    assert summed.mean_aux is False, "summing must stay the default; mean_aux moves every number"
    _, _, aux_s, _ = summed(x, cb, z=z)
    _, _, aux_m, _ = meaned(x, cb, z=z)
    assert torch.allclose(aux_m * 2.0, aux_s, atol=1e-5)      # n_blocks == 2
