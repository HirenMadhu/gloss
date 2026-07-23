"""SetJoin RelBench TEST-set evaluation (leaderboard-comparable) — fork of eval/test_eval.py.

Predict every seed of a split, align to the split table's row order via the per-seed ``input_id``
(carried in the :class:`JoinBatch`), and score with ``task.evaluate``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.graph import coerce_binary_target, make_loader
from .collate import to_join_batch, to_row_set_batch
from .paths import setjoin_neighbors


@torch.no_grad()
def predict_split(
    module,
    bundle,
    task,
    split: str,
    *,
    fanout: int = 64,
    fanout2: int = 0,
    per_relation_cap: int | None = None,
    wide_len: int = 128,
    set_size: int = 128,
    batch_size: int = 128,
    num_workers: int = 0,
    device=None,
) -> np.ndarray:
    """Per-row predictions ``[n]`` aligned to the split table (probability for binary, de-standardized
    value for regression)."""
    device = device or next(module.parameters()).device
    was_training = module.training
    module.eval()
    loader = make_loader(
        bundle, task, split, num_neighbors=setjoin_neighbors(bundle, fanout, fanout2=fanout2),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    n = len(task.get_table(split).df)
    pred = np.full(n, np.nan, dtype=np.float64)
    row_model = getattr(module, "model_arm", "setjoin") == "rowmodel"
    for raw in loader:
        if row_model:
            jb = to_row_set_batch(raw, bundle, task.entity_table,
                                  m_rows=module.m_rows, cells_per_row=module.cells_per_row).to(device)
        else:
            jb = to_join_batch(raw, bundle, task.entity_table,
                               wide_len=wide_len, set_size=set_size, hop2=fanout2 > 0,
                               per_relation_cap=per_relation_cap).to(device)
        out = module(jb).float().cpu()                        # [B] one prediction per seed
        if getattr(module, "task_type", "binary") == "regression":
            pred_b = out * module.target_std + module.target_mean
        else:
            pred_b = torch.sigmoid(out)
        gid = jb.input_id.cpu().numpy()                       # [B] split-table row index per seed
        pred[gid] = pred_b.numpy()
    if was_training:
        module.train()
    assert not np.isnan(pred).any(), f"{split}: {int(np.isnan(pred).sum())} seeds got no prediction"
    return pred


def evaluate_split(
    module,
    bundle,
    task,
    split: str = "test",
    *,
    fanout: int = 64,
    fanout2: int = 0,
    per_relation_cap: int | None = None,
    wide_len: int = 128,
    set_size: int = 128,
    batch_size: int = 128,
    num_workers: int = 0,
    device=None,
) -> dict:
    """Predict ``split`` and score with ``task.evaluate`` (labels coerced for boolean-string targets)."""
    pred = predict_split(
        module, bundle, task, split, fanout=fanout, fanout2=fanout2,
        per_relation_cap=per_relation_cap, wide_len=wide_len,
        set_size=set_size, batch_size=batch_size, num_workers=num_workers, device=device,
    )
    target_table = coerce_binary_target(task.get_table(split, mask_input_cols=False), task)
    return {k: float(v) for k, v in task.evaluate(pred, target_table).items()}
