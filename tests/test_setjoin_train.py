"""SetJoin training sanity: overfit a single rel-f1 batch (binary AND regression), gradients reach
encoder/pool/head end to end, and regression standardization round-trips. Guarded by the cached rel-f1
(repo convention — the full model needs real pytorch-frame stats), mirroring test_train.py.
"""
from __future__ import annotations

import pytest
import torch

from gloss.setjoin.train import SetJoinLitModule
from gloss.train.losses import masked_bce, masked_mse
from gloss.utils.seeding import seed_everything

from ._relf1 import name_table, sample_join_batch
from .conftest import rel_f1_available

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _small_module(bundle, name_emb, entity_table, *, task_type="binary", model_kw=None,
                  **kw) -> SetJoinLitModule:
    seed_everything(0)
    mk = dict(d_model=64, n_heads=4, n_wide_layers=1, n_set_layers=1, n_pma=2, dropout=0.0)
    mk.update(model_kw or {})
    module = SetJoinLitModule(
        bundle, name_emb, entity_table, task_type=task_type, model_kwargs=mk,
        lr=5e-3, wide_len=64, set_size=32, **kw,
    )
    return module.to(_DEVICE)


@rel_f1_available
def test_overfits_single_batch():
    bundle, task, jb = sample_join_batch(batch_size=16)
    jb = jb.to(_DEVICE)
    module = _small_module(bundle, name_table(), task.entity_table)
    assert int(jb.has_target.sum()) > 0, "batch carries no labels to fit"

    opt = torch.optim.AdamW(module.parameters(), lr=5e-3, weight_decay=0.0)
    with torch.no_grad():
        init = masked_bce(module(jb), jb.target, jb.has_target).item()
    final = init
    # 400 steps: with the corrected fanout the batch carries ~10x more set elements than v1, and 200
    # steps left the threshold within CPU-thread-nondeterminism wiggle (observed one flaky failure).
    for _ in range(400):
        opt.zero_grad()
        loss = masked_bce(module(jb), jb.target, jb.has_target)
        loss.backward()
        opt.step()
        final = loss.item()

    assert final < 0.5 * init, f"loss did not drop enough: init={init:.4f} final={final:.4f}"
    assert final < 0.3, f"did not overfit single batch: final={final:.4f}"


@rel_f1_available
@pytest.mark.parametrize("readout_kw", [
    dict(readout="measure"),
    dict(readout="slot", slot_mode="hard"),
    dict(readout="slot", slot_mode="soft"),
    dict(readout="slot", slot_mode="iterative"),
], ids=["measure", "slot-hard", "slot-soft", "slot-iterative"])
def test_readout_arms_overfit_single_batch(readout_kw):
    """End-to-end proof the measure pool reaches the head: each GROUP-BY arm learns a single batch
    (a count-blind no-op — the H2 failure mode — would stall near the initial loss)."""
    bundle, task, jb = sample_join_batch(batch_size=16)
    jb = jb.to(_DEVICE)
    module = _small_module(bundle, name_table(), task.entity_table, model_kw=readout_kw)
    opt = torch.optim.AdamW(module.parameters(), lr=5e-3, weight_decay=0.0)
    with torch.no_grad():
        init = masked_bce(module(jb), jb.target, jb.has_target).item()
    final = init
    for _ in range(250):
        opt.zero_grad()
        loss = masked_bce(module(jb), jb.target, jb.has_target)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.6 * init and final < 0.45, f"{readout_kw}: init={init:.4f} final={final:.4f}"


@rel_f1_available
def test_to_device_carries_set_group_idx():
    """JoinBatch.to() must move AND pass the new field (a forgotten arg TypeErrors on every device move)."""
    bundle, task, jb = sample_join_batch(batch_size=8)
    jb2 = jb.to(_DEVICE)
    assert jb2.set_group_idx.shape == jb.set_group_idx.shape
    assert str(jb2.set_group_idx.device).split(":")[0] == _DEVICE


@rel_f1_available
def test_regression_overfits_single_batch():
    bundle, task, jb = sample_join_batch(batch_size=16)
    jb = jb.to(_DEVICE)
    module = _small_module(bundle, name_table(), task.entity_table, task_type="regression")
    torch.manual_seed(0)
    y = torch.randn(int(jb.num_seeds), device=_DEVICE)
    has = torch.ones_like(jb.has_target)
    opt = torch.optim.AdamW(module.parameters(), lr=5e-3, weight_decay=0.0)
    with torch.no_grad():
        init = masked_mse(module(jb), y, has).item()
    final = init
    for _ in range(200):
        opt.zero_grad()
        loss = masked_mse(module(jb), y, has)
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.3 * init, f"regression did not overfit: init={init:.4f} final={final:.4f}"


@rel_f1_available
def test_gradients_reach_encoder_pool_and_head():
    bundle, task, jb = sample_join_batch(batch_size=8)
    jb = jb.to(_DEVICE)
    module = _small_module(bundle, name_table(), task.entity_table)
    loss = masked_bce(module(jb), jb.target, jb.has_target)
    assert torch.isfinite(loss)
    loss.backward()
    m = module.model
    grads = {
        "name_token": m.encoder.name_proj.weight.grad,
        "value_proj": m.encoder.w_v.weight.grad,
        "row_pool": m.row_pool.w1.weight.grad,
        "set_null": m.set_enc.null_elem.grad,
        "head": m.head[-1].weight.grad,
    }
    for name, g in grads.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


@rel_f1_available
def test_regression_standardization_round_trip():
    bundle, task, _jb = sample_join_batch(batch_size=8)
    module = _small_module(bundle, name_table(), task.entity_table, task_type="regression",
                           target_mean=10.0, target_std=2.5)
    y = torch.tensor([7.5, 10.0, 15.0])
    z = module._standardize(y)
    assert torch.allclose(z, torch.tensor([-1.0, 0.0, 2.0]))
    assert torch.allclose(z * module.target_std + module.target_mean, y)
