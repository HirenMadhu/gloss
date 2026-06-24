"""Proper RelBench TEST-set evaluation (leaderboard-comparable).

RelBench masks test labels in ``get_table('test')`` but ``task.evaluate(pred)`` scores against the
held-out labels internally. We predict every test seed (ordered by ``input_id`` for robustness) and hand
the probability vector to ``task.evaluate`` -> the same metrics (roc_auc, average_precision, ...) the
leaderboard / GelGT table report. Our earlier GATE-1/ablation numbers were VALIDATION; these are TEST.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.collate import to_gloss_batch


def predict_split(forward_fn, bundle, task, split: str, *, num_neighbors, batch_size: int = 512,
                  device="cpu") -> np.ndarray:
    """Run ``forward_fn(gb) -> logits[B]`` over every seed of ``split``; return probabilities ``[n]``
    aligned to the split table's row order (via ``input_id``)."""
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.modeling.loader import CustomNodeLoader, NeighborSampler

    table = task.get_table(split)
    inp = get_node_train_table_input(table, task)
    n = int(inp.nodes[1].shape[0])
    sampler = NeighborSampler(bundle.data, num_neighbors=num_neighbors, time_attr="time",
                              temporal_strategy="last", disjoint=True)
    loader = CustomNodeLoader(bundle.data, node_sampler=sampler, input_nodes=inp.nodes,
                              input_time=inp.time, batch_size=batch_size, shuffle=False)
    pred = np.full(n, np.nan, dtype=np.float64)
    for raw in loader:
        gb = to_gloss_batch(raw, bundle, task.entity_table).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(forward_fn(gb).float()).cpu()        # [B] in segment order
        ent = raw[task.entity_table]
        seg = ent.batch.cpu()                                          # segment per entity node
        gid = ent.input_id.cpu().numpy()                              # global row index per entity node
        pred[gid] = prob[seg].numpy()
    assert not np.isnan(pred).any(), "some test seeds got no prediction"
    return pred


def evaluate_test(forward_fn, bundle, task, *, num_neighbors, batch_size: int = 512, device="cpu") -> dict:
    pred = predict_split(forward_fn, bundle, task, "test", num_neighbors=num_neighbors,
                         batch_size=batch_size, device=device)
    return {k: float(v) for k, v in task.evaluate(pred).items()}
