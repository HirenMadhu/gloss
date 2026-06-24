"""Phase 3 — DOC-RT training sanity: the model can overfit a single rel-f1 batch, and gradients reach
the documentation FiLM, the RT name token, and the head (so the docs pathway is actually trained).

Runs on CUDA when available (the dev node has a GPU), else CPU. Guarded by the cached rel-f1 dataset.
"""
from __future__ import annotations

import torch

from gloss.train.loop import DOCRTLitModule
from gloss.train.losses import masked_bce
from gloss.utils.seeding import seed_everything

from ._relf1 import groundings, sample_cell_batch
from .conftest import rel_f1_available

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _small_module(bundle, grounding, entity_table) -> DOCRTLitModule:
    seed_everything(0)
    module = DOCRTLitModule(
        bundle, grounding, entity_table,
        model_kwargs=dict(d_model=64, d_text=grounding.d_text, n_blocks=2,
                          n_heads=4, d_ff=128, enc_channels=64),
        lr=5e-3, seq_len=256, max_fk=5,
    )
    return module.to(_DEVICE)


@rel_f1_available
def test_overfits_single_batch():
    """Repeatedly fitting one batch must drive the masked BCE far below its random-init value."""
    bundle, task, cb = sample_cell_batch(seq_len=256, batch_size=16)
    cb = cb.to(_DEVICE)
    g_full, _g_null, _g_name = groundings()
    module = _small_module(bundle, g_full, task.entity_table)
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
def test_gradients_reach_docs_and_head():
    """One backward pass must deliver finite, non-zero gradients to the FiLM(γ), the RT name token, and
    the head — i.e. the documentation-conditioning pathway is differentiable end to end."""
    bundle, task, cb = sample_cell_batch(seq_len=256, batch_size=8)
    cb = cb.to(_DEVICE)
    g_full, _g_null, _g_name = groundings()
    module = _small_module(bundle, g_full, task.entity_table)

    loss = masked_bce(module(cb), cb.target, cb.has_target)
    assert torch.isfinite(loss)
    loss.backward()

    enc = module.model.encoder
    grads = {
        "film_gamma": enc.gamma.weight.grad,
        "name_token": enc.name_proj.weight.grad,
        "head": next(module.model.head.parameters()).grad,
    }
    for name, g in grads.items():
        assert g is not None, f"no gradient reached {name}"
        assert torch.isfinite(g).all(), f"non-finite gradient at {name}"
        assert g.abs().sum() > 0, f"zero gradient at {name}"
