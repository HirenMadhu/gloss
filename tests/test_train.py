"""RT training sanity: the model can overfit a single rel-f1 batch, and gradients reach the RT name
token, the value projection, and the head (so the model is trained end to end).

Runs on CUDA when available (the dev node has a GPU), else CPU. Guarded by the cached rel-f1 dataset.
"""
from __future__ import annotations

import torch

from gloss.text.cache import HashEncoder
from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none
from gloss.train.loop import MoRELitModule
from gloss.train.losses import masked_bce, masked_mse
from gloss.utils.seeding import seed_everything

from ._relf1 import name_table, sample_cell_batch
from .conftest import rel_f1_available

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _small_module(bundle, name_emb, entity_table, *, task_type="binary",
                  device=None) -> MoRELitModule:
    seed_everything(0)
    module = MoRELitModule(
        bundle, name_emb, entity_table, task_type=task_type,
        model_kwargs=dict(d_model=64, n_blocks=2, n_heads=4, d_ff=128, enc_channels=64,
                          table_name_emb=build_table_name_embeddings(bundle, HashEncoder(dim=64)),
                          role_name_emb=role_name_embeddings_with_none(bundle, HashEncoder(dim=64))),
        lr=5e-3, seq_len=256, max_fk=5,
    )
    return module.to(device or _DEVICE)


@rel_f1_available
def test_overfits_single_batch():
    """Repeatedly fitting one batch must drive the masked BCE far below its random-init value."""
    bundle, task, cb = sample_cell_batch(seq_len=256, batch_size=16)
    cb = cb.to(_DEVICE)
    name_emb = name_table()
    module = _small_module(bundle, name_emb, task.entity_table)
    assert int(cb.has_target.sum()) > 0, "batch carries no labels to fit"

    opt = torch.optim.AdamW(module.parameters(), lr=5e-3, weight_decay=0.0)
    with torch.no_grad():
        init = masked_bce(module(cb), cb.target, cb.has_target).item()
    final = init
    for _ in range(200):
        opt.zero_grad()
        loss = masked_bce(module(cb), cb.target, cb.has_target)
        loss.backward()
        opt.step()
        final = loss.item()

    assert final < 0.5 * init, f"loss did not drop enough: init={init:.4f} final={final:.4f}"
    assert final < 0.3, f"did not overfit single batch: final={final:.4f}"


@rel_f1_available
def test_gradients_reach_encoder_and_head():
    """One backward pass must deliver finite, non-zero gradients to the RT name token, the value
    projection, and the head — i.e. the model is differentiable end to end."""
    bundle, task, cb = sample_cell_batch(seq_len=256, batch_size=8)
    cb = cb.to(_DEVICE)
    name_emb = name_table()
    module = _small_module(bundle, name_emb, task.entity_table)

    loss = masked_bce(module(cb), cb.target, cb.has_target)
    assert torch.isfinite(loss)
    loss.backward()

    enc = module.model.encoder
    grads = {
        "name_token": enc.name_proj.weight.grad,
        "value_proj": enc.w_v.weight.grad,
        "head": next(module.model.head.parameters()).grad,
    }
    for name, g in grads.items():
        assert g is not None, f"no gradient reached {name}"
        assert torch.isfinite(g).all(), f"non-finite gradient at {name}"
        assert g.abs().sum() > 0, f"zero gradient at {name}"


@rel_f1_available
def test_regression_overfits_single_batch():
    """The regression path (continuous target via masked MSE) can memorize one batch.

    Pinned to **CPU**, unlike the rest of this module, and the reason is measured. This test used to
    fail about one run in three, from two independent causes:

    1. *The fixture, not the model.* ``HashTextEmbedder`` seeded its per-string RNG with Python's
       builtin ``hash()``, which is randomized per process, so ``sample_cell_batch`` materialized a
       different rel-f1 every run. Fixed at the source in ``gloss/data/graph.py``.
    2. *The optimization does not survive CUDA nondeterminism.* ``RowPool``'s ``index_add``
       reductions use CUDA atomics (its own docstring says so), giving ~1.5e-7 between two identical
       forwards. Compounded over a few hundred AdamW steps that is not last-bit noise, it is a
       different basin: with the fixture fixed and ``init`` identical to 4 decimal places, three
       sequential CUDA trials of this exact fit reached 0.5431 / 0.0000 / 1.1412 at step 200, and the
       third was still at 0.9655 at step 600. ``seed_everything(deterministic_torch=True)`` does not
       help — it passes ``warn_only=True`` and the offending scatter has no deterministic CUDA
       kernel to fall back to.

    On CPU the same fit is bit-reproducible (0.000004 at step 200 in three consecutive trials, five
    orders of magnitude under the threshold) and *faster* than the CUDA path, 16 s against 34 s, since
    the model is deliberately tiny. The binary sibling above stays on CUDA: it converges to 0.0000 by
    step 200 in every trial, so it has no margin to lose.
    """
    bundle, task, cb = sample_cell_batch(seq_len=256, batch_size=16)
    cb = cb.to("cpu")
    name_emb = name_table()
    module = _small_module(bundle, name_emb, task.entity_table, task_type="regression",
                           device="cpu")
    torch.manual_seed(0)
    y = torch.randn(int(cb.num_seeds))                         # synthetic continuous target
    has = torch.ones_like(cb.has_target)
    opt = torch.optim.AdamW(module.parameters(), lr=5e-3, weight_decay=0.0)
    with torch.no_grad():
        init = masked_mse(module(cb), y, has).item()
    final = init
    for _ in range(300):
        opt.zero_grad()
        loss = masked_mse(module(cb), y, has)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.3 * init, f"regression did not overfit: init={init:.4f} final={final:.4f}"
