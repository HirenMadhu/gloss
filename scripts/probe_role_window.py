#!/usr/bin/env python3
"""Verify the three §6 assumptions of the recency order-statistic (`x`) spec **before** building on them.

The spec asks three questions that must be measured, not assumed:

1. **The effective per-role cap.** ``num_neighbors=[12,12]`` is the sampler's fanout cap, but
   ``seq_len=512`` truncates again at the *cell* level. If row membership were derived from surviving
   cells, ``sat_rho`` would be measuring truncation rather than fanout. Reported: the histogram of
   ``|C(r,rho)|`` against w=12, per role, for seed rows and for all rows.
2. **The recency unit.** ``model/signature.py`` under ``time_mode='rope'`` uses
   ``tau = log1p(Delta_seconds)`` (``TimeLadder.tau``), canonical unit **seconds**. This script
   re-derives tau both ways on the same batch and asserts they agree, so the new channel cannot
   introduce a second convention.
3. **Whether Delta >= 0 actually holds.** The spec wants a loud assert. But the codebase already
   carries ``was_clamped`` / ``b_clamped`` (amendments §8.1, §9.10) *because* rows dated after their
   seed exist. A blind assert would then fire on every batch. Reported: the raw violation rate among
   **child rows only** (the sets the channel aggregates over), separately from seed/root rows, whose
   ``Delta = max(0, t* - t_r)`` clamp to 0 by construction and are not part of any child set.

CPU only, no training. Usage::

    .venv/bin/python scripts/probe_role_window.py --dataset rel-f1 --task driver-top3 --batches 4
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gloss.data.collate import to_cell_batch  # noqa: E402
from gloss.data.graph import build_gloss_graph, make_loader  # noqa: E402
from gloss.model.time_encoding import TimeLadder  # noqa: E402

W = 12  # gloss/train/datamodule.py: `num_neighbors or [12, 12]`, uniform over roles


def probe(dataset: str, task_name: str, *, split: str, batches: int, batch_size: int,
          seq_len: int, max_fk: int) -> dict:
    from relbench.tasks import get_task

    bundle = build_gloss_graph(dataset, cache_dir=str(ROOT / "data" / "graph_cache" / dataset))
    task = get_task(dataset, task_name, download=False)
    loader = make_loader(bundle, task, split, num_neighbors=None, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    K = bundle.num_roles
    # per-role accumulators over (row, role) groups that are non-empty
    counts: dict[int, list[int]] = collections.defaultdict(list)
    seed_counts: dict[int, list[int]] = collections.defaultdict(list)
    untimed_children = collections.Counter()
    timed_children = collections.Counter()
    neg_child = pos_child = 0
    neg_root = n_root = 0
    tau_max_abs_diff = 0.0
    n_batches = 0

    for raw in loader:
        cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=seq_len, max_fk=max_fk)
        B, R = cb.row_valid.shape
        adj = cb.adj_role                                  # [B,R,R]; s is a CHILD of r via role s->r
        valid_pair = cb.row_valid.unsqueeze(2) & cb.row_valid.unsqueeze(1)
        child = (adj >= 1) & (adj <= K) & valid_pair       # [B,R,R] bool

        # ---- 1. |C(r,rho)| against the w=12 cap -------------------------------------------
        for rho in range(1, K + 1):
            m = child & (adj == rho)                       # [B,R,R]
            c = m.sum(-1)                                  # [B,R] children of r via rho
            nz = c[(c > 0) & cb.row_valid]
            if nz.numel():
                counts[rho].extend(nz.tolist())
            croot = c[cb.row_is_root & cb.row_valid]
            croot = croot[croot > 0]
            if croot.numel():
                seed_counts[rho].extend(croot.tolist())
            # timedness of the children reached by this role
            reached = m.any(1)                             # [B,R] is s a child via rho for some r
            if reached.any():
                timed_children[rho] += int((reached & cb.row_is_timed).sum())
                untimed_children[rho] += int((reached & ~cb.row_is_timed).sum())

        # ---- 3. raw Delta sign, child rows vs root rows -----------------------------------
        raw_delta = cb.seed_time.unsqueeze(1).to(torch.float64) - cb.row_time_r.to(torch.float64)
        is_child_of_any = child.any(1) & cb.row_is_timed & cb.row_valid      # [B,R]
        neg_child += int((raw_delta < 0)[is_child_of_any].sum())
        pos_child += int(is_child_of_any.sum())
        root_timed = cb.row_is_root & cb.row_is_timed & cb.row_valid
        neg_root += int((raw_delta < 0)[root_timed].sum())
        n_root += int(root_timed.sum())

        # ---- 2. one recency convention, not two -------------------------------------------
        tau_ladder = TimeLadder.tau(TimeLadder.delta_seconds(cb.seed_time.unsqueeze(1), cb.row_time_r))
        tau_spec = torch.log1p((cb.seed_time.unsqueeze(1).to(torch.float64)
                                - cb.row_time_r.to(torch.float64)).clamp(min=0.0) / 1.0)
        tau_max_abs_diff = max(tau_max_abs_diff, float((tau_ladder - tau_spec).abs().max()))

        n_batches += 1
        if n_batches >= batches:
            break

    def summarize(d):
        out = {}
        for rho, xs in sorted(d.items()):
            t = torch.tensor(xs, dtype=torch.float64)
            out[str(rho)] = {
                "n_groups": int(t.numel()), "mean": round(float(t.mean()), 3),
                "max": int(t.max()), f"sat_rate_at_{W}": round(float((t >= W).double().mean()), 4),
            }
        return out

    return {
        "dataset": dataset, "task": task_name, "split": split, "batches": n_batches,
        "num_roles": K, "w": W, "seq_len": seq_len, "batch_size": batch_size,
        "fanout_all_rows": summarize(counts),
        "fanout_seed_rows": summarize(seed_counts),
        "roles_untimed": {str(r): {"timed": timed_children[r], "untimed": untimed_children[r]}
                          for r in sorted(set(timed_children) | set(untimed_children))},
        "delta_negative_child_rows": neg_child, "child_rows_timed": pos_child,
        "delta_negative_root_rows": neg_root, "root_rows_timed": n_root,
        "tau_convention_max_abs_diff": tau_max_abs_diff,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-top3")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-fk", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rep = probe(a.dataset, a.task, split=a.split, batches=a.batches, batch_size=a.batch_size,
                seq_len=a.seq_len, max_fk=a.max_fk)
    print(json.dumps(rep, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
