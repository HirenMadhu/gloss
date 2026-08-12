"""measure_substrate.py — re-derive the two load-bearing numbers `changes.md` cites from `report.md`.

`report.md` is absent from the repo, so its figures are unverifiable. Two of them are load-bearing:

* **Post-truncation rows per seed** — `changes.md` P0.2 sets ``R = 160`` as a hard *assert*, so a wrong
  value fails a run outright rather than degrading it. Report claims maxima rel-f1 69 / rel-trial 63 /
  rel-stack 139.
* **Masked/padded attention fraction** — the entire justification for Phase 0b collapsing the four
  masked attentions into one. Report claims 97–98% of attention FLOPs land on masked or padded pairs.

Both are measured off the real collate path, no model and no name embeddings required.

    .venv/bin/python scripts/measure_substrate.py --datasets rel-f1 --batches 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.to(torch.float64)
    q = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)
    p50, p90, p99 = torch.quantile(x, q).tolist()
    return {"mean": x.mean().item(), "p50": p50, "p90": p90, "p99": p99, "max": x.max().item()}


def measure(dataset: str, *, batches: int, seq_len: int, batch_size: int,
            num_neighbors: list[int], split: str,
            omega: tuple[float, float] = (0.05, 5.0), n_freq: int = 8) -> dict:
    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.model.rt_substrate import build_relational_masks
    from gloss.eval.ablation import dataset_tasks

    bundle = build_gloss_graph(dataset)
    # leaderboard tasks only, matching the standing rule for what this repo runs;
    # fall back to any entity task so a dataset absent from LEADERBOARD_TASKS still measures
    tasks = dataset_tasks([dataset], ["leaderboard"]).get(dataset, [])
    if not tasks:
        tasks = dataset_tasks([dataset]).get(dataset, [])
        if tasks:
            print(f"  [{dataset}: not in LEADERBOARD_TASKS; falling back to entity task {tasks[0]!r}]")
    if not tasks:
        raise SystemExit(f"no entity tasks for {dataset}")
    task_name = tasks[0]

    from relbench.tasks import get_task

    task = get_task(dataset, task_name, download=True)
    entity_table = task.entity_table
    loader = make_loader(bundle, task, split, num_neighbors=num_neighbors,
                         batch_size=batch_size, shuffle=False)

    rows_per_seed: list[torch.Tensor] = []
    cells_per_seed: list[torch.Tensor] = []
    mask_true = {k: 0.0 for k in ("col", "feat", "nbr", "full")}
    pad_true = 0.0
    n_pairs = 0.0
    n_seeds = 0
    taus: list[torch.Tensor] = []   # log1p(Delta) over real, timed cells — changes.md 3.1 / 9.8
    n_real_cells = 0
    n_untimed_cells = 0

    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        cb = to_cell_batch(batch, bundle, entity_table, seq_len=seq_len)
        real = ~cb.is_padding                                     # [B, S]
        B, S = real.shape

        # rows per seed = distinct node_idxs among real cells
        for b in range(B):
            nid = cb.node_idxs[b][real[b]]
            rows_per_seed.append(torch.tensor(float(nid.unique().numel())))
            cells_per_seed.append(torch.tensor(float(int(real[b].sum()))))
        n_seeds += B

        # tau over real cells: Delta in SECONDS against the seed time, tau = log1p(Delta).
        timed = real & cb.is_timed
        n_real_cells += int(real.sum())
        n_untimed_cells += int((real & ~cb.is_timed).sum())
        if bool(timed.any()):
            delta = (cb.seed_time.unsqueeze(1) - cb.row_time).clamp_min(0.0)
            taus.append(torch.log1p(delta[timed].to(torch.float64)))

        masks = build_relational_masks(cb)
        for k, m in masks.items():
            mask_true[k] += float(m.sum())
        pad_pair = real.unsqueeze(2) & real.unsqueeze(1)
        pad_true += float(pad_pair.sum())
        n_pairs += float(B * S * S)

    rows = torch.stack(rows_per_seed)
    cells = torch.stack(cells_per_seed)

    dens = {k: v / n_pairs for k, v in mask_true.items()}
    mean_dens = sum(dens.values()) / len(dens)

    # changes.md 9.8: the ladder constants [0.05, 5.0] x n_freq=8 are asserted from the claimed
    # tau in [0, 22] range, never measured. A channel whose sin(omega*tau) has ~zero variance is
    # outside the data's spread and the band should move; the lowest channel must not wrap
    # (omega_min * tau_span < pi) or it stops being monotonic in recency.
    tau_stats = None
    if taus:
        tau = torch.cat(taus)
        omegas = torch.logspace(float(torch.log10(torch.tensor(omega[0]))),
                                float(torch.log10(torch.tensor(omega[1]))), n_freq,
                                dtype=torch.float64)
        theta = tau.unsqueeze(1) * omegas.unsqueeze(0)              # [N, n_freq]
        span = float(tau.max() - tau.min())
        tau_stats = {
            "mean": tau.mean().item(), "std": tau.std().item(),
            "min": tau.min().item(), "max": tau.max().item(), "span": span,
            "q": _quantiles(tau),
            "omegas": omegas.tolist(),
            "sin_var": torch.sin(theta).var(dim=0).tolist(),
            "lowest_wraps": omegas[0].item() * span >= torch.pi,
            "frac_untimed": n_untimed_cells / max(n_real_cells, 1),
            "in_band_0_22": bool(tau.min() >= 0.0 and tau.max() <= 22.0),
        }

    return {
        "dataset": dataset, "task": task_name, "split": split, "seeds": n_seeds,
        "seq_len": seq_len, "num_neighbors": num_neighbors,
        "rows": _quantiles(rows), "cells": _quantiles(cells),
        "pad_pair_density": pad_true / n_pairs,
        "mask_density": dens, "mean_mask_density": mean_dens,
        "wasted_frac": 1.0 - mean_dens,
        "tau": tau_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-neighbors", type=int, nargs="+", default=[12, 12])
    ap.add_argument("--split", default="train")
    ap.add_argument("--omega", type=float, nargs=2, default=[0.05, 5.0],
                    help="RoPE ladder band to audit against the observed tau (changes.md 3.1)")
    ap.add_argument("--n-freq", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    for ds in args.datasets:
        try:
            r = measure(ds, batches=args.batches, seq_len=args.seq_len,
                        batch_size=args.batch_size, num_neighbors=args.num_neighbors,
                        split=args.split, omega=tuple(args.omega), n_freq=args.n_freq)
        except Exception as e:  # keep going; one dataset failing must not hide the others
            print(f"\n=== {ds}: FAILED — {type(e).__name__}: {e}")
            continue
        print(f"\n=== {ds} / {r['task']} ({r['seeds']} seeds, seq_len={r['seq_len']}, "
              f"fanout={r['num_neighbors']})")
        rq, cq = r["rows"], r["cells"]
        print(f"  rows/seed   mean {rq['mean']:7.1f}  p50 {rq['p50']:6.0f}  p90 {rq['p90']:6.0f}  "
              f"p99 {rq['p99']:6.0f}  MAX {rq['max']:6.0f}")
        print(f"  cells/seed  mean {cq['mean']:7.1f}  p50 {cq['p50']:6.0f}  p90 {cq['p90']:6.0f}  "
              f"p99 {cq['p99']:6.0f}  MAX {cq['max']:6.0f}   (truncation cap {r['seq_len']})")
        d = r["mask_density"]
        print(f"  mask density  col {d['col']:.4f}  feat {d['feat']:.4f}  nbr {d['nbr']:.4f}  "
              f"full {d['full']:.4f}   (pad-pair {r['pad_pair_density']:.4f})")
        print(f"  mean density {r['mean_mask_density']:.4f}  ->  "
              f"WASTED {r['wasted_frac']*100:.2f}% of the 4 x S^2 score matrix")
        t = r["tau"]
        if t:
            tq = t["q"]
            print(f"  tau=log1p(dt)  mean {t['mean']:6.2f}  std {t['std']:5.2f}  "
                  f"min {t['min']:6.2f}  p50 {tq['p50']:6.2f}  p99 {tq['p99']:6.2f}  "
                  f"max {t['max']:6.2f}   untimed {t['frac_untimed']*100:.1f}%")
            chans = "  ".join(f"{w:.3f}:{v:.3f}" for w, v in zip(t["omegas"], t["sin_var"]))
            print(f"  ladder var(sin w*tau)  {chans}")
            dead = [w for w, v in zip(t["omegas"], t["sin_var"]) if v < 0.01]
            flags = []
            if not t["in_band_0_22"]:
                flags.append("tau OUTSIDE [0,22] -> dt is not in seconds")
            if t["lowest_wraps"]:
                flags.append(f"lowest omega WRAPS over span {t['span']:.1f}")
            if dead:
                flags.append(f"{len(dead)} dead channel(s) (var<0.01): "
                             + ", ".join(f"{w:.3f}" for w in dead))
            print(f"  ladder verdict: {'; '.join(flags) if flags else 'band OK for this dataset'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
