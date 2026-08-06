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


# ---------------------------------------------------------------------------------------------
# TWO-LEVEL hyperparameter grid (changes.md `arch: two_level`).
#
# Deliberately SMALL and targeted, not a cross product for its own sake. The full-design run beat
# RT on driver-top3 (+7 AUROC) but LOST on driver-dnf (-8) and driver-position (NMAE 0.629 vs 0.430),
# so the hypothesis under test is a CAPACITY / LR mismatch rather than a broken mechanism:
#   * the earlier RT arch grid found small `d_model=128` best, while the two-level runs used 256;
#   * a two-level block has SIX sublayers to RT's five, so n_blocks=8 is 48 sublayers — likely too
#     deep. Sweep 2 and 4.
#   * more parameters at the same lr can simply be undertrained in 10 epochs; sweep lr.
# 2 x 2 x 2 = 8 configs. n_heads=8 divides both d_model values.
TWO_LEVEL_D_MODEL = [128, 256]
TWO_LEVEL_N_BLOCKS = [2, 4]
TWO_LEVEL_LR = [3.0e-4, 1.0e-3]

#: `extended` adds 1e-4 BELOW the current range. Motivated twice over: our own grid found 3e-4 far
#: better than 1e-3 (33/69 cells beat RT vs 15/71), so the optimum may lie below the swept range;
#: and GelGT trains at 1e-4. It is a SEPARATE set rather than an edit to TWO_LEVEL_LR because
#: changing that list re-maps every array index — arrays already queued would silently run a
#: different experiment (amendments.md §9.3).
#: `scaled` is for LARGE-BATCH runs. Going 64 -> 512 is 8x, and the optimum lr moves UP with batch
#: size (linear scaling says 8x -> 2.4e-3, sqrt scaling 2.8x -> 8.5e-4). It keeps 3e-4 as the
#: unscaled control so "the lr moved" and "big batches help" stay separable.
#: `single` pins the one lr that was val-selected on the most tasks (6 of 9) in the b512/80ep
#: multi-seed run. It exists so a MECHANISM ablation can spend its whole array on arms rather than on
#: an lr axis: the arms are compared as a paired delta at a fixed config, where the lr is a held
#: constant, not a thing being selected.
LR_SETS = {"default": TWO_LEVEL_LR, "extended": [1.0e-4, 3.0e-4, 1.0e-3], "single": [3.0e-4],
           "scaled": [3.0e-4, 1.0e-3, 3.0e-3]}

#: Swept `(d_model, n_blocks)` pairs. `default` MUST stay equal to
#: `product(TWO_LEVEL_D_MODEL, TWO_LEVEL_N_BLOCKS)` or every queued array's index->job map shifts
#: (amendments.md §9.3); `tests/test_runner_cli.py` pins that.
#:
#: `small` is 128/2 alone — the ONLY config `scripts/probe_batch_size.py` measured as fitting batch
#: 512 without gradient accumulation (38.3 GiB; 256/4 OOMs above 256). That restriction is not a
#: budget compromise, it is required for the experiment to mean anything on the binary tasks: the
#: pairwise AUC surrogate forms its pairs WITHIN a batch, so accum=2 at batch 256 would leave the
#: pair count at 256's level while looking like 512 in the record.
GRID_SETS = {"default": list(itertools.product(TWO_LEVEL_D_MODEL, TWO_LEVEL_N_BLOCKS)),
             "small": [(128, 2)]}


def two_level_grid(lr_set: str = "default", grid_set: str = "default") -> list[dict]:
    try:
        lrs = LR_SETS[lr_set]
    except KeyError:
        raise ValueError(f"unknown lr_set {lr_set!r}; expected one of {sorted(LR_SETS)}") from None
    try:
        shapes = GRID_SETS[grid_set]
    except KeyError:
        raise ValueError(f"unknown grid_set {grid_set!r}; expected one of {sorted(GRID_SETS)}") from None
    grid = []
    for (dm, nb), lr in itertools.product(shapes, lrs):
        grid.append(dict(d_model=dm, n_blocks=nb, n_heads=8, d_ff=dm * 4,
                         num_experts=4, enc_channels=dm, lr=lr))
    return grid


def arch_grid() -> list[dict]:
    grid = []
    for dm, nb, nh, ffm, ne, enc in itertools.product(
        D_MODEL, N_BLOCKS, N_HEADS, D_FF_MULT, NUM_EXPERTS, ENC_CHANNELS
    ):
        grid.append(dict(d_model=dm, n_blocks=nb, n_heads=nh,
                         d_ff=dm * ffm, num_experts=ne, enc_channels=enc))
    return grid


#: Task subsets selectable with `--tasks`. `regression` is the 4 NMAE-scored entity tasks — the only
#: ones a change to the regression objective can affect, so an L1/Huber sweep runs just these.
TASK_SETS = {
    "all": TASKS,
    "regression": [("rel-f1", "driver-position"), ("rel-trial", "study-adverse"),
                   ("rel-trial", "site-success"), ("rel-event", "user-attendance")],
    "binary": [t for t in TASKS if t not in
               {("rel-f1", "driver-position"), ("rel-trial", "study-adverse"),
                ("rel-trial", "site-success"), ("rel-event", "user-attendance")}],
}


def jobs(seeds: int, arch: str = "rt", task_set: str = "all",
         lr_set: str = "default", grid_set: str = "default") -> list[tuple]:
    """Flat (config_idx, cfg, dataset, task, seed) list the array iterates by index.

    `task_set` / `lr_set` / `grid_set` all change the index->job mapping, so each MUST be passed
    identically to `--list` and to `run_index` — see the §9.3 comment in `main`. Results for
    different settings therefore belong in different `--out-dir`s: index 0 means a different job
    under each.
    """
    g = two_level_grid(lr_set, grid_set) if arch == "two_level" else arch_grid()
    try:
        tasks = TASK_SETS[task_set]
    except KeyError:
        raise ValueError(f"unknown task_set {task_set!r}; expected one of {sorted(TASK_SETS)}") from None
    return [(ci, cfg, ds, tk, s)
            for ci, cfg in enumerate(g)
            for (ds, tk) in tasks
            for s in range(seeds)]


def router_diagnostics(module, bundle, task, *, seq_len: int, max_fk: int,
                       batch_size: int, n_batches: int = 4) -> dict:
    """Expert usage / entropy / column-partition size for the trained router, from a few VAL batches.

    Wired in because the method's central claim — that routing on the relational signature
    *specialises* the experts — has never once been measured (`results/WHY_NOT_RT.md` §2): the
    diagnostics existed but nothing called them, and the grid saves no checkpoints, so every finished
    run so far is unprobeable after the fact. If usage has collapsed onto one expert then MoRE is RT
    with a wider FFN and no other number here means much.

    VAL, not train (routing on data the model fit tells you less) and not test (which we only touch
    once, through `task.evaluate`). Failures are recorded, never raised: a diagnostic must not be able
    to destroy the run's actual result.
    """
    import torch

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import make_loader
    from gloss.eval.diagnostics import expert_usage, mean_active_experts, specialization_probe

    out: dict = {}
    try:
        device = next(module.parameters()).device
        loader = make_loader(bundle, task, "val", num_neighbors=None,
                             batch_size=batch_size, shuffle=False, num_workers=0)
        cbs = []
        for raw in loader:
            cbs.append(to_cell_batch(raw, bundle, task.entity_table,
                                     seq_len=seq_len, max_fk=max_fk).to(device))
            if len(cbs) >= n_batches:
                break
    except Exception as exc:
        return {"router_error": f"batches: {exc!r}"}

    # Each probe is guarded SEPARATELY. Under one shared try/except a single stale probe returned
    # `router_error` for the whole record and silently took the two working ones with it — which is
    # exactly what a CPU smoke run caught before these arrays launched.
    def _probe(key, fn):
        try:
            return fn()
        except Exception as exc:
            out[f"{key}_error"] = repr(exc)
            return None

    with torch.no_grad():
        usage_ent = _probe("router_usage", lambda: expert_usage(module.model, cbs))
        kbar = _probe("router_mean_active_k", lambda: mean_active_experts(module.model, cbs))
        part = _probe("router_specialization", lambda: specialization_probe(module.model))

    if usage_ent is not None and usage_ent[0] is not None:
        usage, entropy = usage_ent
        u = usage.tolist()
        n_exp = len(u)
        out.update(router_usage=u, router_entropy=entropy,
                   # uniform usage has entropy log(E); 1.0 = balanced, ->0 = collapsed onto one
                   router_entropy_norm=entropy / math.log(n_exp) if n_exp > 1 else float("nan"),
                   router_max_usage=max(u))
    elif usage_ent is not None:
        out["router_usage_error"] = "no MoEFFN in this model"
    if kbar is not None:
        out["router_mean_active_k"] = kbar
    if part is not None:
        # how many DISTINCT experts the columns partition into: 1 == the router learned nothing
        # column-specific, whatever the aggregate gate mass looks like
        out["router_columns_probed"] = len(part)
        out["router_distinct_experts"] = len(set(part.values())) if part else 0
    return out


def x_channel_diagnostics(module, bundle, task, *, seq_len: int, max_fk: int,
                          batch_size: int, n_batches: int = 4) -> dict:
    """§5 instrumentation for the recency order-statistic channel, on **val**.

    The two numbers that matter are invisible in the accuracy column:

    * ``x_kappa_max`` drifting toward 0 means the soft max collapsed to a **mean** — no order
      statistic is being computed and the arm silently stopped testing the hypothesis. Compare
      against the init (+4.0 / -4.0), which is what makes drift readable from a single final value.
    * ``x_alpha`` staying near 0 means the channel is unused and the arm is a no-op.

    Wrapped per-probe like ``router_diagnostics``: a diagnostic must never take down a finished run.
    """
    import torch

    try:
        model = module.model
        ch = getattr(getattr(model, "substrate", None), "x_channel", None)
        if ch is None:
            return {"x_channel_error": "no x_channel on this model"}
        device = next(module.parameters()).device
        loader = make_loader(bundle, task, "val", num_neighbors=None,
                             batch_size=batch_size, shuffle=False, num_workers=0)
        out: dict = {}
        with torch.no_grad():
            for raw in loader:
                cb = to_cell_batch(raw, bundle, task.entity_table,
                                   seq_len=seq_len, max_fk=max_fk).to(device)
                _, out = ch(cb)
                break
        for k in ("x_kappa_max", "x_kappa_min", "x_alpha"):
            out[k] = float(getattr(ch, k.replace("x_", "")).detach())
        out["x_kappa_max_init"], out["x_kappa_min_init"] = 4.0, -4.0
        return out
    except Exception as exc:
        return {"x_channel_error": repr(exc)}


def init_batch(cfg: dict) -> int:
    """Per-config starting batch (MoE dense-combine runs all experts on all tokens -> ~M x FFN mem)."""
    heavy = cfg["d_model"] * cfg["n_blocks"] * cfg["num_experts"] * cfg["d_ff"]
    if heavy >= 32_000_000:
        return 16
    if heavy >= 16_000_000:
        return 32
    return 64


def run_index(index, *, seeds, epochs, num_workers, seq_len, max_fk, out_dir,
              arch: str = "rt", phase: str = "full", encoder: str = "qwen",
              task_set: str = "all", regression_loss: str = "mse",
              binary_loss: str = "bce", lr_set: str = "default",
              optimizer: str = "adamw", weight_decay: float = 0.01,
              target_scaling: str = "zscore", clamp_pct: float | None = None,
              batch_size: int | None = None, accum: int = 1,
              grid_set: str = "default", patience: int = 3,
              recency_channel: str = "off") -> dict:
    import numpy as np
    import torch

    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.test_eval import evaluate_split, score_pred
    from gloss.train.finetune import name_embeddings, target_stats, task_kind, train_prebuilt

    J = jobs(seeds, arch, task_set, lr_set, grid_set)
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
    # ENCODER must match whatever the run being compared against used. The two-level array used
    # qwen, so a harrier grid would not be comparable to it — this used to be hardcoded to harrier.
    d_text = 5376 if encoder == "harrier" else 2560
    name_emb = name_embeddings(bundle, ds, encoder=encoder, d_text=d_text)   # cached; no model load
    kind = task_kind(task)
    mk = dict(cfg)
    mk["k"] = 2
    lr = mk.pop("lr", 3.0e-4)          # lr is a TRAINER arg, not a model kwarg
    if arch == "two_level":
        from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none
        from gloss.train.finetune import _name_encoder
        from run_ablation_phases import TWO_LEVEL_PHASES

        enc = _name_encoder(ds, encoder=encoder, d_text=d_text)
        mk.update(arch="two_level",
                  table_name_emb=build_table_name_embeddings(bundle, enc, kind="query"),
                  role_name_emb=role_name_embeddings_with_none(bundle, enc, kind="query"),
                  **TWO_LEVEL_PHASES[phase])
        mk["recency_channel"] = recency_channel

    bs = batch_size or init_batch(cfg)
    module = metrics = None
    while True:
        try:
            module, metrics = train_prebuilt(
                bundle, task, name_emb, model_kwargs=mk, route_on="signature", lambda_ortho=0.5,
                seq_len=seq_len, max_fk=max_fk, batch_size=bs, max_epochs=epochs,
                seed=seed, num_workers=num_workers, lr=lr, regression_loss=regression_loss,
                binary_loss=binary_loss, optimizer=optimizer, weight_decay=weight_decay,
                target_scaling=target_scaling, clamp_pct=clamp_pct, accumulate_grad_batches=accum,
                patience=patience,
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or bs <= 8:
                raise
            torch.cuda.empty_cache()
            bs = max(8, bs // 2)
            print(f"CUDA OOM -> retry with batch_size={bs}", flush=True)

    rec = {"config_idx": ci, "dataset": ds, "task": tk, "seed": seed,
           "task_type": kind, "batch_size": bs, "arch": arch, "encoder": encoder,
           # stamp the objective on the record: an L1 run and an MSE run are otherwise
           # indistinguishable from their JSON, and they live in sibling directories
           "regression_loss": regression_loss if kind == "regression" else None,
           "binary_loss": binary_loss if kind == "binary" else None,
           "lr_set": lr_set, "optimizer": optimizer, "weight_decay": weight_decay,
           "target_scaling": target_scaling if kind == "regression" else None,
           "clamp_pct": clamp_pct if kind == "regression" else None, "accum": accum,
           # epochs/patience used to be un-stamped, so a 10-epoch run and an 80-epoch run were
           # indistinguishable from their JSON — the same gap the objective stamps above closed.
           # A large-batch arm is defined as much by its epoch budget as by its batch size.
           "epochs": epochs, "patience": patience, "grid_set": grid_set,
           "recency_channel": recency_channel,
           "task_set": task_set, **cfg, "k": 2}
    if arch == "two_level":
        rec["phase"] = phase
    rec.update({f"val_{k.split('/')[-1]}": v for k, v in metrics.items() if k.startswith("val/")})
    rec.update(router_diagnostics(module, bundle, task, seq_len=seq_len, max_fk=max_fk,
                                  batch_size=bs))
    if recency_channel != "off":
        rec.update(x_channel_diagnostics(module, bundle, task, seq_len=seq_len, max_fk=max_fk,
                                         batch_size=bs))
    try:
        # Score UNCLAMPED and keep the raw predictions, then apply the clamp to the same array. One
        # trained model yields both numbers, so the clamp's eval effect is isolated from any change
        # in what was learned; a second training run would confound the two.
        tm, pred = evaluate_split(module, bundle, task, "test", num_neighbors=None,
                                  seq_len=seq_len, max_fk=max_fk, batch_size=bs,
                                  num_workers=num_workers, clamp=False, return_pred=True)
        rec.update({f"test_{k}": v for k, v in tm.items()})
        std = target_stats(task)[1] if kind == "regression" else None
        if kind == "regression" and rec.get("test_mae") is not None:
            rec["test_nmae"] = rec["test_mae"] / std
        np.save(out_dir / f"{index:05d}_pred.npy", pred)
        if kind == "regression" and clamp_pct:
            from gloss.train.finetune import target_clamp

            lo, hi = target_clamp(task, clamp_pct, 100.0 - clamp_pct)
            cm = score_pred(task, "test", np.clip(pred, lo, hi))
            rec.update({f"test_clamped_{k}": v for k, v in cm.items()})
            rec["clamp_bounds"] = [lo, hi]
            if cm.get("mae") is not None:
                rec["test_clamped_nmae"] = cm["mae"] / std
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
    ap.add_argument("--arch", default="rt", choices=["rt", "two_level"],
                    help="two_level sweeps TWO_LEVEL_GRID (d_model x n_blocks x lr), not arch_grid")
    ap.add_argument("--phase", default="full", choices=["phase0a", "phase0b", "full"])
    ap.add_argument("--encoder", default="qwen",
                    help="MUST match the run you compare against (the two-level array used qwen)")
    ap.add_argument("--tasks", default="all", choices=sorted(TASK_SETS),
                    help="task subset; CHANGES the index->job mapping, so use a distinct --out-dir")
    ap.add_argument("--reg-loss", default="mse", choices=["mse", "l1", "huber"],
                    help="regression objective (binary tasks ignore it). mse = every result before "
                         "2026-07-31; l1 aligns training with RelBench's MAE metric")
    ap.add_argument("--lr-set", default="default", choices=sorted(LR_SETS),
                    help="extended adds lr=1e-4 (12 configs, not 8); scaled adds 3e-3 for "
                         "large-batch runs. CHANGES index->job: new --out-dir")
    ap.add_argument("--grid-set", default="default", choices=sorted(GRID_SETS),
                    help="swept (d_model, n_blocks) pairs; small = 128/2 only, the config that fits "
                         "batch 512 unaccumulated. CHANGES index->job: new --out-dir")
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stop patience in EPOCHS. Scale it with --epochs when the batch "
                         "grows: at 8x the batch an epoch is 1/8 the steps, so patience 3 would "
                         "stop 8x earlier in step terms and undo the point of the longer budget")
    # The recency order-statistic arms. This does NOT change the index->job map (it is not part of
    # `jobs()`), so each arm needs its own --out-dir, not a shared one: index 0 means the same
    # (config, task, seed) under every arm and the skip-if-done guard keys on index alone.
    ap.add_argument("--recency-channel", default="off",
                    choices=["off", "full", "flags", "shuffle"],
                    help="row-level recency order-statistic channel: off=base | full=the mechanism | "
                         "flags=saturation/exists/untimed only (is the win the age, or just knowing "
                         "it truncated?) | shuffle=real features, Delta permuted across seeds (placebo)")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"],
                    help="adam + --weight-decay 1e-5 reproduces GelGT's optimizer")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="0.01 = ours, 1e-5 = GelGT")
    ap.add_argument("--target-scaling", default="zscore", choices=["zscore", "raw"],
                    help="regression only; raw trains on the unstandardized target (GelGT does)")
    ap.add_argument("--clamp-pct", type=float, default=None,
                    help="regression only; clip preds to the train target's [p, 100-p] percentiles "
                         "(GelGT uses 2). Applied to val for selection; test is scored BOTH ways")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the per-config heuristic (still halves on CUDA OOM)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation; equivalent to a bigger batch for bce/mse/l1 but "
                         "NOT for the pairwise auc loss (its pairs are within-batch)")
    ap.add_argument("--bin-loss", default="bce", choices=["bce", "auc"],
                    help="binary objective (regression tasks ignore it). bce = every result before "
                         "2026-07-31; auc is a pairwise squared-hinge AUROC surrogate")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    if args.list:
        print(len(jobs(args.seeds, args.arch, args.tasks, args.lr_set, args.grid_set)))
        return 0
    if args.aggregate:
        print(aggregate(out_dir))
        return 0
    if args.index is None:
        print("pass --index N, --list, or --aggregate")
        return 2
    # arch/phase/encoder MUST be forwarded. They used to be parsed and then silently dropped here, so
    # `run_index` fell back to its defaults (arch="rt", encoder="qwen") no matter what was on the
    # command line — while `--list` above DID honour `args.arch`. That mismatch is what made it
    # invisible: the array was sized 72 from the two-level grid but every task ran the 864-entry RT
    # grid, and a `--encoder harrier` array quietly ran qwen. 96 completed GPU-jobs of the wrong
    # experiment (amendments.md §9.3) plus a wasted harrier cache build. Keep this call exhaustive.
    rec = run_index(args.index, seeds=args.seeds, epochs=args.epochs, num_workers=args.num_workers,
                    seq_len=args.seq_len, max_fk=args.max_fk, out_dir=out_dir,
                    arch=args.arch, phase=args.phase, encoder=args.encoder,
                    task_set=args.tasks, regression_loss=args.reg_loss,
                    binary_loss=args.bin_loss, lr_set=args.lr_set, optimizer=args.optimizer,
                    weight_decay=args.weight_decay, target_scaling=args.target_scaling,
                    clamp_pct=args.clamp_pct, batch_size=args.batch_size, accum=args.accum,
                    grid_set=args.grid_set, patience=args.patience,
                    recency_channel=args.recency_channel)
    print({k: rec.get(k) for k in ("config_idx", "dataset", "task", "seed", "task_type",
                                   "batch_size", "test_roc_auc", "test_nmae")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
