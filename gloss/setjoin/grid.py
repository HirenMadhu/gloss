"""SetJoin backbone sweep (phase S10): signature vs signature+shared × size, SLURM-array friendly.

The fairness counterpart of MoRE's architecture grid search (`scripts/run_gridsearch.py`): MoRE's
compare-table numbers are per-task winners of a ~90-config sweep while SetJoin has run ONE untuned
backbone — this grid gives the single-table substrate the same treatment. 18 configs = the full
product d_model {128,256} × depth {2,4} (n_wide=n_set) × d_ff {2×,4×} × {base, +shared expert}
plus two d512 probes (d512/depth2/ff×2, both arms — over the ~30M param budget, report-only;
kept because MoRE's rel-event winners were d512). Every config pins ``enc_channels =
min(d_model, 256)`` so the pytorch-frame stype encoders never scale past what MoRE's sweep ran.

One (config, task, seed) per array index -> one ``{index:05d}.json``; idempotent by existence.
``aggregate_grid`` prints per-task winners vs the RT-from-scratch baseline AND MoRE's grid best,
plus a per-config mean-rank leaderboard with the ≤30M "adopted backbone" line (the standing config
for phases S11/S12 must respect the repo's param budget; over-budget probes are flagged).
"""
from __future__ import annotations

import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from ..utils.paths import REPO
from .runner import MORE_GRID_BEST, attach_nmae

# Order fixed (defines array indexing) — do not reorder once an array is launched.
TASKS = [
    ("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3"), ("rel-f1", "driver-position"),
    ("rel-trial", "study-outcome"), ("rel-trial", "study-adverse"), ("rel-trial", "site-success"),
    ("rel-event", "user-repeat"), ("rel-event", "user-ignore"), ("rel-event", "user-attendance"),
]

D_MODEL = [128, 256]
DEPTH = [2, 4]                      # n_wide_layers = n_set_layers
D_FF_MULT = [2, 4]                  # d_ff = d_model * mult
SHARED = [False, True]
D512_PROBES = [(512, 2, 2, False), (512, 2, 2, True)]   # report-only (36M params > ~30M budget)

PARAM_BUDGET = 30e6

# every key below is a SetJoin.__init__ kwarg — the whole cfg IS model_kwargs
FIXED = dict(route_on="signature", num_experts=4, k=2, d_sig=64, n_heads=4, n_pma=4)


def _cfg(dm: int, depth: int, mult: int, shared: bool) -> dict:
    return dict(d_model=dm, n_wide_layers=depth, n_set_layers=depth, d_ff=dm * mult,
                use_shared=shared, enc_channels=min(dm, 256), **FIXED)


def arch_grid() -> list[dict]:
    grid = [_cfg(dm, dep, mult, sh)
            for dm, dep, mult, sh in itertools.product(D_MODEL, DEPTH, D_FF_MULT, SHARED)]
    grid += [_cfg(*probe) for probe in D512_PROBES]
    return grid


def jobs(seeds: int) -> list[tuple]:
    """Flat (config_idx, cfg, dataset, task, seed) list the array iterates by index."""
    return [(ci, cfg, ds, tk, s)
            for ci, cfg in enumerate(arch_grid())
            for (ds, tk) in TASKS
            for s in range(seeds)]


def init_batch(cfg: dict) -> int:
    """Starting batch: dense-combine MoE runs all experts (+shared) on all tokens every layer."""
    depth = cfg["n_wide_layers"] + cfg["n_set_layers"]
    heavy = depth * cfg["d_model"] * cfg["d_ff"] * (cfg["num_experts"] + int(cfg["use_shared"]))
    return 128 if heavy < 8_000_000 else 64


def run_index(index: int, *, seeds: int, epochs: int, num_workers: int, out_dir: Path,
              limit_train_batches=None, limit_val_batches=None) -> dict:
    import torch
    from relbench.tasks import get_task

    from ..data.graph import build_gloss_graph
    from ..train.finetune import name_embeddings, target_stats, task_kind
    from ..utils.paths import graph_cache_dir
    from .eval import evaluate_split
    from .train import train_prebuilt_setjoin

    J = jobs(seeds)
    if index >= len(J):
        print(f"index {index} >= grid size {len(J)}; nothing to do")
        return {}
    ci, cfg, ds, tk, seed = J[index]
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / f"{index:05d}.json"
    if done_path.exists():                       # idempotent resume
        print(f"index {index} already done ({done_path.name}); skip")
        return json.loads(done_path.read_text())

    bundle = build_gloss_graph(ds, cache_dir=str(graph_cache_dir(ds)))
    task = get_task(ds, tk, download=False)
    name_emb = name_embeddings(bundle, ds, encoder="harrier", d_text=2560)   # cached; no LM load
    kind = task_kind(task)

    bs = init_batch(cfg)
    module = metrics = None
    while True:
        try:
            module, metrics = train_prebuilt_setjoin(
                bundle, task, name_emb, model_kwargs=dict(cfg),
                fanout=64, wide_len=128, set_size=128, lambda_ortho=0.5,
                batch_size=bs, lr=3e-4, weight_decay=0.01, max_epochs=epochs,
                seed=seed, num_workers=num_workers,
                limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    rec = {"config_idx": ci, "dataset": ds, "task": tk, "seed": seed, "task_type": kind,
           "batch_size": bs, "n_params": sum(p.numel() for p in module.model.parameters()), **cfg}
    rec.update({f"val_{k.split('/')[-1]}": v for k, v in metrics.items() if k.startswith("val/")})
    try:
        tm = evaluate_split(module, bundle, task, "test", fanout=64, wide_len=128, set_size=128,
                            batch_size=bs, num_workers=num_workers)
        rec.update({f"test_{k}": v for k, v in tm.items()})
        attach_nmae(rec, target_stats(task)[1])
    except Exception as exc:                     # keep the trained val metrics if test eval fails
        rec["test_error"] = repr(exc)
    done_path.write_text(json.dumps(rec))
    return rec


CFG_KEYS = ("d_model", "n_wide_layers", "n_set_layers", "d_ff", "use_shared", "enc_channels")


def pick_backbone(out_dir: Path) -> tuple[int, dict]:
    """The S10 adoption rule, machine-readable: best mean-rank config with max n_params ≤ 30M
    (the sweep contains the v3 default as cfg with d128/2+2/ff256, so 'keep the default' is the
    automatic outcome when nothing bigger out-ranks it). Raises if no records qualify."""
    PRIMARY = {"binary": "test_roc_auc", "regression": "test_nmae"}
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        grp[(r["dataset"], r["task"], r["config_idx"])].append(r)
    by_task: dict[tuple, list] = defaultdict(list)
    params_of: dict[int, int] = {}
    for (ds, tk, ci), rows in grp.items():
        key = PRIMARY[rows[0]["task_type"]]
        xs = [r[key] for r in rows
              if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not xs:
            continue
        params_of[ci] = max(params_of.get(ci, 0), int(rows[0].get("n_params", 0)))
        by_task[(ds, tk)].append((ci, st.mean(xs), rows[0]["task_type"]))
    n_cfg = len(arch_grid())
    ranks: dict[int, list[int]] = defaultdict(list)
    for (ds, tk), cands in by_task.items():
        lower = cands[0][2] == "regression"
        ranked = sorted(cands, key=lambda c: (c[1] if lower else -c[1]))
        seen = set()
        for pos, c in enumerate(ranked):
            ranks[c[0]].append(pos + 1)
            seen.add(c[0])
        for ci in range(n_cfg):
            if ci not in seen:
                ranks[ci].append(len(ranked) + 1)
    board = sorted((st.mean(rs), ci) for ci, rs in ranks.items()
                   if rs and params_of.get(ci, 0) <= PARAM_BUDGET)
    if not board:
        raise RuntimeError(f"no ≤30M config with records in {out_dir}")
    ci = board[0][1]
    return ci, arch_grid()[ci]


def _cfg_str(cfg: dict) -> str:
    return (f"d{cfg['d_model']}x{cfg['n_wide_layers']}+{cfg['n_set_layers']} ff{cfg['d_ff']}"
            f"{' +shared' if cfg['use_shared'] else ''}")


def aggregate_grid(out_dir: Path) -> str:
    PRIMARY = {"binary": "test_roc_auc", "regression": "test_nmae"}
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    if not recs:
        return "(no records)"
    rt_path = REPO / "results" / "rt_from_scratch_baseline.json"
    rt = json.loads(rt_path.read_text())["tasks"] if rt_path.exists() else {}

    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        grp[(r["dataset"], r["task"], r["config_idx"])].append(r)

    by_task: dict[tuple, list] = defaultdict(list)
    params_of: dict[int, int] = {}
    for (ds, tk, ci), rows in grp.items():
        ttype = rows[0]["task_type"]
        key = PRIMARY[ttype]
        xs = [r[key] for r in rows
              if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not xs:
            continue
        params_of[ci] = max(params_of.get(ci, 0), int(rows[0].get("n_params", 0)))
        cfg = {k: rows[0][k] for k in CFG_KEYS}
        by_task[(ds, tk)].append((ci, st.mean(xs), len(xs), ttype, key, cfg))

    lines = [f"{len(recs)} runs over {len(grp)} (task,config) cells.", ""]
    beats_rt = beats_more = 0
    ranks: dict[int, list[int]] = defaultdict(list)
    n_cfg = len(arch_grid())
    for (ds, tk) in TASKS:
        cands = by_task.get((ds, tk))
        if not cands:
            continue
        ttype, key = cands[0][3], cands[0][4]
        lower = key == "test_nmae"
        ranked = sorted(cands, key=lambda c: (c[1] if lower else -c[1]))
        seen = set()
        for pos, c in enumerate(ranked):
            ranks[c[0]].append(pos + 1)
            seen.add(c[0])
        for ci in range(n_cfg):                          # missing config -> worst rank + 1
            if ci not in seen:
                ranks[ci].append(len(ranked) + 1)
        ci, m, n, _, _, cfg = ranked[0]
        rtv = rt.get(f"{ds}/{tk}", {}).get("value")
        more = MORE_GRID_BEST.get((ds, tk))
        ours = m * 100 if ttype == "binary" else m
        beat_rt = rtv is not None and ((ours > rtv) if ttype == "binary" else (ours < rtv))
        beat_more = more is not None and ((ours > more) if ttype == "binary" else (ours < more))
        beats_rt += bool(beat_rt)
        beats_more += bool(beat_more)
        unit = "AUROC↑" if ttype == "binary" else "NMAE↓"
        over = "  [>30M]" if params_of.get(ci, 0) > PARAM_BUDGET else ""
        lines.append(f"=== {ds}/{tk}  ({unit}) ===")
        lines.append(f"  best cfg#{ci}: {_cfg_str(cfg)}  ({params_of.get(ci, 0)/1e6:.1f}M{over})")
        lines.append(f"  ours={ours:.4f} (n={n})   RT={rtv}  {'BEAT✅' if beat_rt else 'no❌'}   "
                     f"MoRE-best={more}  {'BEAT✅' if beat_more else 'no❌'}")
        for ci2, m2, n2, _, _, cfg2 in ranked[1:4]:
            disp = m2 * 100 if ttype == "binary" else m2
            lines.append(f"    runner-up cfg#{ci2}: {disp:.4f}  ({_cfg_str(cfg2)})")
        lines.append("")

    grid_cfgs = arch_grid()
    board = sorted((st.mean(rs), ci) for ci, rs in ranks.items() if rs)
    lines.append(f"Per-config mean rank across {len(by_task)} tasks (top 5):")
    for mr, ci in board[:5]:
        over = "  [>30M]" if params_of.get(ci, 0) > PARAM_BUDGET else ""
        lines.append(f"  cfg#{ci}: mean rank {mr:.2f}  {_cfg_str(grid_cfgs[ci])}  "
                     f"({params_of.get(ci, 0)/1e6:.1f}M{over})")
    if board:
        mr, ci = board[0]
        lines.append(f"winner (unrestricted): cfg#{ci} {_cfg_str(grid_cfgs[ci])} (mean rank {mr:.2f})")
        ok = [(mr2, ci2) for mr2, ci2 in board if params_of.get(ci2, 0) <= PARAM_BUDGET]
        if ok:
            mr, ci = ok[0]
            lines.append(f"ADOPTED BACKBONE (best mean-rank ≤30M): cfg#{ci} "
                         f"{_cfg_str(grid_cfgs[ci])} (mean rank {mr:.2f}, "
                         f"{params_of.get(ci, 0)/1e6:.1f}M)")
    lines.append(f"beats RT: {beats_rt}/{len(by_task)}   beats MoRE grid best: "
                 f"{beats_more}/{len(by_task)}")
    lines.append(f"(best-of-{n_cfg} per task — winner's curse; confirm any adopted config "
                 "with a clean re-run before headline claims)")
    return "\n".join(lines)
