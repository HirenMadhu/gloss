"""RelBench TEST-set evaluation for DOC-RT (leaderboard-comparable).

RelBench masks the labels in ``task.get_table('test')`` (only the seed keys + time survive), but
``task.evaluate(pred)`` scores a prediction vector against the held-out labels it keeps internally.
We therefore predict *every* seed of a split, align the probabilities to the split table's row order
via the entity store's ``input_id``, and hand the vector to ``task.evaluate``.

The headline GATE (``eval/ablation.py``) reports **validation** metrics — the correct, non-leaky split
for the docs-on-vs-off comparison. This module produces the separate, leaderboard-style **test** number
(it requires the trained module in memory, since the gate saves no checkpoints).
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.collate import to_cell_batch
from ..data.graph import make_loader


@torch.no_grad()
def predict_split(
    module,
    bundle,
    task,
    split: str,
    *,
    num_neighbors,
    seq_len: int,
    max_fk: int,
    batch_size: int = 64,
    num_workers: int = 0,
    device=None,
) -> np.ndarray:
    """Run ``module(cb) -> logits[B]`` over every seed of ``split``; return probabilities ``[n]``
    aligned to the split table's row order (via the entity store's ``input_id``)."""
    device = device or next(module.parameters()).device
    was_training = module.training
    module.eval()
    loader = make_loader(
        bundle, task, split, num_neighbors=num_neighbors,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    n = len(task.get_table(split).df)
    pred = np.full(n, np.nan, dtype=np.float64)
    for raw in loader:
        cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=seq_len, max_fk=max_fk).to(device)
        prob = torch.sigmoid(module(cb).float()).cpu()        # [B] in segment order
        store = raw[task.entity_table]
        seg = store.batch.cpu()                               # segment per entity node
        gid = store.input_id.cpu().numpy()                    # row index per entity node (seed row)
        pred[gid] = prob[seg].numpy()
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
    num_neighbors,
    seq_len: int,
    max_fk: int,
    batch_size: int = 64,
    num_workers: int = 0,
    device=None,
) -> dict:
    """Predict ``split`` and score with ``task.evaluate`` (RelBench metrics: average_precision,
    roc_auc, accuracy, f1). For ``test`` the labels are internal (pass ``target_table=None``); for
    other splits the table carries labels."""
    pred = predict_split(
        module, bundle, task, split, num_neighbors=num_neighbors,
        seq_len=seq_len, max_fk=max_fk, batch_size=batch_size, num_workers=num_workers, device=device,
    )
    target_table = None if split == "test" else task.get_table(split)
    return {k: float(v) for k, v in task.evaluate(pred, target_table).items()}
