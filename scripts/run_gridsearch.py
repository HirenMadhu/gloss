"""run_gridsearch.py — architecture grid search on the signature arm (SLURM-array friendly).

Sweeps (d_model, n_blocks, n_heads, d_ff, enc_channels, num_experts) x {the 9 RT-reported entity tasks}
x seeds, all ``route_on=signature`` with the cached harrier schema embeddings. One (config, task, seed)
per array index -> one JSON. ``--aggregate`` picks the best config per task (seed-mean of the primary
metric) and compares to the saved RT (from scratch) baseline.

    python scripts/run_gridsearch.py --list                        # array size
    python scripts/run_gridsearch.py --index $SLURM_ARRAY_TASK_ID  # run one (config, task, seed)
    python scripts/run_gridsearch.py --aggregate                   # best config per task vs RT

Fixed knobs (not swept): route_on=signature, k=2, lambda_ortho=0.5, seq_len=512, encoder=harrier.
Batch size starts at a per-config heuristic and halves on CUDA OOM (floor 8).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]

# The 9 entity tasks RT (from scratch) reports on the leaderboard (rel-f1 / rel-trial / rel-event).
# Order is fixed (it defines the array indexing) — do not reorder once an array is launched.
TASKS = [
    ("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3"), ("rel-f1", "driver-position"),
    ("rel-trial", "study-outcome"), ("rel-trial", "study-adverse"), ("rel-trial", "site-success"),
    ("rel-event", "user-repeat"), ("rel-event", "user-ignore"), ("rel-event", "user-attendance"),
]

# Swept architecture axes. n_heads in {4,8} both divide every d_model here, so all combos are valid.
D_MODEL = [128, 256, 512]
N_BLOCKS = [4, 8]
N_HEADS = [4, 8]
D_FF_MULT = [2, 4]          # d_ff = d_model * mult
NUM_EXPERTS = [4, 8]
ENC_CHANNELS = [128, 256]


def arch_grid() -> list[dict]:
    grid = []
    for dm, nb, nh, ffm, ne, enc in itertools.product(
        D_MODEL, N_BLOCKS, N_HEADS, D_FF_MULT, NUM_EXPERTS, ENC_CHANNELS
    ):
        grid.append(dict(d_model=dm, n_blocks=nb, n_heads=nh,
                         d_ff=dm * ffm, num_experts=ne, enc_channels=enc))
    return grid


def jobs(seeds: int) -> list[tuple]:
    """Flat (config_idx, cfg, dataset, task, seed) list the array iterates by index."""
    return [(ci, cfg, ds, tk, s)
            for ci, cfg in enumerate(arch_grid())
            for (ds, tk) in TASKS
            for s in range(seeds)]


def init_batch(cfg: dict) -> int:
    """Per-config starting batch (MoE dense-combine runs all experts on all tokens -> ~M x FFN mem)."""
    heavy = cfg["d_model"] * cfg["n_blocks"] * cfg["num_experts"] * cfg["d_ff"]
    if heavy >= 32_000_000:
        return 16
    if heavy >= 16_000_000:
        return 32
    return 64


def run_index(index, *, seeds, epochs, num_workers, seq_len, max_fk, out_dir) -> dict:
    import torch

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.test_eval import evaluate_split
    from gloss.train.finetune import name_embeddings, target_stats, task_kind, train_prebuilt

    J = jobs(seeds)
    if index >= len(J):
        print(f"index {index} >= grid size {len(J)}; nothing to do")
        return {}
    ci, cfg, ds, tk, seed = J[index]
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"{index:05d}.json"
    if done_path.exists():                       # idempotent resume: skip already-finished configs
        print(f"index {index} already done ({done_path.name}); skip")
        return {}
    graph_cache = str(REPO / "data" / "graph_cache" / ds)
    bundle = build_gloss_graph(ds, cache_dir=graph_cache)
    task = get_task(ds, tk, download=False)
    name_emb = name_embeddings(bundle, ds, encoder="harrier", d_text=2560)   # cached; no model load
    kind = task_kind(task)
    mk = dict(cfg)
    mk["k"] = 2

    bs = init_batch(cfg)
    module = metrics = None
    while True:
        try:
            module, metrics = train_prebuilt(
                bundle, task, name_emb, model_kwargs=mk, route_on="signature", lambda_ortho=0.5,
                seq_len=seq_len, max_fk=max_fk, batch_size=bs, max_epochs=epochs,
                seed=seed, num_workers=num_workers,
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    rec = {"config_idx": ci, "dataset": ds, "task": tk, "seed": seed,
           "task_type": kind, "batch_size": bs, **cfg, "k": 2}
    rec.update({f"val_{k.split('/')[-1]}": v for k, v in metrics.items() if k.startswith("val/")})
    try:
        tm = evaluate_split(module, bundle, task, "test", num_neighbors=None,
                            seq_len=seq_len, max_fk=max_fk, batch_size=bs, num_workers=num_workers)
        rec.update({f"test_{k}": v for k, v in tm.items()})
        if kind == "regression" and rec.get("test_mae") is not None:
            rec["test_nmae"] = rec["test_mae"] / target_stats(task)[1]
    except Exception as exc:                       # keep the (trained) val metrics if test eval fails
        rec["test_error"] = repr(exc)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{index:05d}.json").write_text(json.dumps(rec))
    return rec


def aggregate(out_dir: Path) -> str:
    PRIMARY = {"binary": "test_roc_auc", "regression": "test_nmae"}
    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not recs:
        return "(no records)"
    from gloss.eval.leaderboard import METHODS, beats, load as load_baselines
    base = load_baselines()

    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        grp[(r["dataset"], r["task"], r["config_idx"])].append(r)

    by_task: dict[tuple, list] = defaultdict(list)
    for (ds, tk, ci), rows in grp.items():
        ttype = rows[0]["task_type"]
        key = PRIMARY[ttype]
        xs = [r[key] for r in rows
              if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not xs:
            continue
        cfg = {k: rows[0][k] for k in ("d_model", "n_blocks", "n_heads", "d_ff", "enc_channels", "num_experts")}
        by_task[(ds, tk)].append((ci, st.mean(xs), len(xs), ttype, key, cfg))

    lines = [f"{len(recs)} runs over {len(grp)} (task,config) cells.", ""]
    wins = {name: 0 for name in METHODS}          # per-baseline win counts
    scored = {name: 0 for name in METHODS}        # tasks that baseline actually reports
    for (ds, tk) in TASKS:
        cands = by_task.get((ds, tk))
        if not cands:
            continue
        ttype, key = cands[0][3], cands[0][4]
        lower = key == "test_nmae"
        ranked = sorted(cands, key=lambda c: (c[1] if lower else -c[1]))
        ci, m, n, _, _, cfg = ranked[0]
        entry = base.get(f"{ds}/{tk}", {"type": ttype})
        # regression records already carry NMAE (test_mae / train-std), so both types are in
        # leaderboard units after the binary x100.
        ours_disp = m * 100 if ttype == "binary" else m
        unit = "AUROC↑" if ttype == "binary" else "NMAE↓"
        cmp_parts = []
        for name in METHODS:
            bv = entry.get(name)
            won = beats(ours_disp, bv, ttype)
            if won is None:
                cmp_parts.append(f"{name}=n/a")
                continue
            scored[name] += 1
            wins[name] += won
            cmp_parts.append(f"{name}={bv}  {'BEAT ✅' if won else 'no ❌'}")
        lines.append(f"=== {ds}/{tk}  ({unit}) ===")
        lines.append(f"  best cfg#{ci}: d_model={cfg['d_model']} n_blocks={cfg['n_blocks']} "
                     f"n_heads={cfg['n_heads']} d_ff={cfg['d_ff']} enc={cfg['enc_channels']} "
                     f"experts={cfg['num_experts']}")
        lines.append(f"  ours={ours_disp:.4f} (n={n})   " + "   ".join(cmp_parts))
        for ci2, m2, n2, _, _, cfg2 in ranked[1:3]:
            disp = m2 * 100 if ttype == "binary" else m2
            lines.append(f"    runner-up cfg#{ci2}: {disp:.4f}  "
                         f"(dm{cfg2['d_model']} nb{cfg2['n_blocks']} nh{cfg2['n_heads']} "
                         f"ff{cfg2['d_ff']} enc{cfg2['enc_channels']} e{cfg2['num_experts']})")
        lines.append("")
    for name in METHODS:
        lines.append(f"Best-config beats {name} on {wins[name]}/{scored[name]} reported tasks.")
    lines.append(f"(NOTE: best-of-{len(arch_grid())} per task — optimistic; confirm winners with a clean re-run.)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print array size and exit")
    ap.add_argument("--aggregate", action="store_true", help="best config per task vs RT")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-fk", type=int, default=5)
    ap.add_argument("--out-dir", default="results/gridsearch")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    if args.list:
        print(len(jobs(args.seeds)))
        return 0
    if args.aggregate:
        print(aggregate(out_dir))
        return 0
    if args.index is None:
        print("pass --index N, --list, or --aggregate")
        return 2
    rec = run_index(args.index, seeds=args.seeds, epochs=args.epochs, num_workers=args.num_workers,
                    seq_len=args.seq_len, max_fk=args.max_fk, out_dir=out_dir)
    print({k: rec.get(k) for k in ("config_idx", "dataset", "task", "seed", "task_type",
                                   "batch_size", "test_roc_auc", "test_nmae")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
