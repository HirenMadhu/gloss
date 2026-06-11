"""Phase 4 — the LightGBM floor baseline (no graph encoder, no docs).

Per seed, aggregate the numeric features of its sampled-subgraph rows into a fixed-width vector
(per node type: a node count + the column-wise mean of that type's numerical cells), then fit a
gradient-boosted tree. This is the 'is the relational/doc machinery even needed?' floor — deliberately
independent of HALOS. NOT the thesis (that's HALOS-full vs HALOS-null); just an absolute reference.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.graph import GraphBundle
from .metrics import binary_metrics_prob


def _numeric_cols(bundle: GraphBundle):
    """Per node type: number of numerical feature columns (fixed per DB -> fixed feature width)."""
    import torch_frame

    out = {}
    for nt in bundle.node_types:
        cnd = bundle.data[nt].tf.col_names_dict
        out[nt] = len(cnd.get(torch_frame.numerical, []))
    return out


def seed_features(loader, bundle: GraphBundle, entity_table: str):
    """Iterate a loader -> (X [n_seeds, F], y [n_seeds]). Seeds without labels are dropped."""
    import torch_frame

    ncols = _numeric_cols(bundle)
    types = [nt for nt in bundle.node_types]
    # feature layout: per type -> [count, mean_0..mean_{c-1}]
    offsets, F = {}, 0
    for nt in types:
        offsets[nt] = F
        F += 1 + ncols[nt]

    Xs, ys = [], []
    for raw in loader:
        ent = raw[entity_table]
        B = int(max(raw[nt].batch.max() for nt in raw.node_types if raw[nt].num_nodes > 0)) + 1
        feat = np.zeros((B, F), dtype=np.float32)
        for nt in types:
            if nt not in raw.node_types or raw[nt].num_nodes == 0:
                continue
            seg = raw[nt].batch.cpu().numpy()
            counts = np.bincount(seg, minlength=B)
            feat[:, offsets[nt]] = counts
            c = ncols[nt]
            if c == 0:
                continue
            num = raw[nt].tf.feat_dict.get(torch_frame.numerical)
            if num is None:
                continue
            x = torch.nan_to_num(num.float(), nan=0.0).cpu().numpy()      # [n, c]
            summ = np.zeros((B, c), dtype=np.float32)
            np.add.at(summ, seg, x)
            denom = np.clip(counts, 1, None)[:, None]
            feat[:, offsets[nt] + 1: offsets[nt] + 1 + c] = summ / denom
        # labels
        if "y" not in ent:
            continue
        y = ent.y.cpu().numpy().astype(int)
        yb = ent.batch.cpu().numpy()[: y.shape[0]] if y.shape[0] != ent.batch.numel() else ent.batch.cpu().numpy()
        yfull = np.full(B, -1, dtype=int)
        yfull[yb] = y
        keep = yfull >= 0
        Xs.append(feat[keep])
        ys.append(yfull[keep])
    return np.concatenate(Xs), np.concatenate(ys)


def run_lightgbm_baseline(bundle, task, *, num_neighbors=None, batch_size=512, n_estimators=200):
    """Train LightGBM on train-split aggregated features, evaluate on val. -> metrics dict."""
    import lightgbm as lgb

    from ..data.graph import make_loader

    num_neighbors = num_neighbors or [12, 12]
    tr = make_loader(bundle, task, "train", num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False)
    va = make_loader(bundle, task, "val", num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False)
    Xtr, ytr = seed_features(tr, bundle, task.entity_table)
    Xva, yva = seed_features(va, bundle, task.entity_table)
    clf = lgb.LGBMClassifier(n_estimators=n_estimators, verbosity=-1)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xva)[:, 1]
    m = binary_metrics_prob(prob, yva)
    m["n_train"], m["n_val"], m["n_features"] = len(ytr), len(yva), Xtr.shape[1]
    return m
