"""bench_complexity.py — wall-clock/memory microbenchmark of the two substrates on one GPU.

Times model compute only (forward, and forward+backward) on ONE fixed real minibatch per substrate —
sampling/collate cost is excluded (both substrates use the same relbench NeighborLoader family).
Configs mirror what actually trained: MoRE at its per-task gridsearch-winner backbones (seq_len 512,
batch 64), SetJoin at the gate defaults (wide_len 128, set_size 128, batch 128; also timed at 64 for
a same-batch comparison). Reports per-seed latency so different batch sizes stay comparable.

    .venv/bin/python scripts/bench_complexity.py [--dataset rel-f1 --task driver-dnf]
    # -> prints a table; writes results/complexity/bench_<gpu>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _bench(fn, *, warmup: int = 5, iters: int = 20) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _measure(model, batch, label: str, B: int, rows: list[dict]) -> None:
    import torch

    model.cuda().train()
    n_params = sum(p.numel() for p in model.parameters())

    def fwd():
        with torch.no_grad():
            model(batch)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        logits, aux = model(batch)
        (logits.float().sum() + aux).backward()

    t_f = _bench(fwd)
    torch.cuda.reset_peak_memory_stats()
    t_fb = _bench(fwd_bwd)
    mem = torch.cuda.max_memory_allocated() / 2**30
    rows.append(dict(label=label, batch=B, params_m=n_params / 1e6,
                     fwd_ms=t_f * 1e3, fwd_bwd_ms=t_fb * 1e3,
                     fwd_bwd_ms_per_seed=t_fb * 1e3 / B, peak_mem_gb=mem))
    print(f"{label:44s} B={B:<4d} params={n_params/1e6:6.2f}M  fwd={t_f*1e3:7.2f}ms  "
          f"fwd+bwd={t_fb*1e3:7.2f}ms  ({t_fb*1e3/B:5.2f}ms/seed)  peak={mem:5.2f}GB", flush=True)
    model.cpu()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-dnf")
    ap.add_argument("--encoder", default="harrier")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--wide-len", type=int, default=128)
    ap.add_argument("--set-size", type=int, default=128)
    ap.add_argument("--fanout", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import torch
    from relbench.tasks import get_task

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.model.more import MoRE
    from gloss.setjoin.collate import to_join_batch
    from gloss.setjoin.model import SetJoin
    from gloss.setjoin.paths import setjoin_neighbors
    from gloss.train.finetune import name_embeddings
    from gloss.utils.paths import graph_cache_dir
    from gloss.utils.seeding import seed_everything

    assert torch.cuda.is_available(), "benchmark needs a GPU"
    seed_everything(0)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}\n")

    bundle = build_gloss_graph(args.dataset, cache_dir=str(graph_cache_dir(args.dataset)))
    task = get_task(args.dataset, args.task, download=False)
    d_text = 64 if args.encoder == "hash" else 2560
    name_emb = name_embeddings(bundle, args.dataset, encoder=args.encoder, d_text=d_text)
    ent = task.entity_table
    rows: list[dict] = []

    # ---- MoRE (multi-table cell-token substrate): gridsearch loader settings ----
    for B in (64,):
        loader = make_loader(bundle, task, "train", num_neighbors=[12, 12], batch_size=B,
                             shuffle=False)
        cb = to_cell_batch(next(iter(loader)), bundle, ent, seq_len=args.seq_len, max_fk=5).to("cuda")
        cells = int((~cb.is_padding).sum())
        print(f"MoRE CellBatch: B={B} S={args.seq_len} (real cells {cells}, "
              f"{cells/B:.0f}/seed)")
        # the two rel-f1 gridsearch-winner backbones (driver-dnf / driver-top3)
        for tag, kw in (
            ("more[signature 128x8 h8 ff256 e4]", dict(d_model=128, n_blocks=8, n_heads=8,
                                                       d_ff=256, num_experts=4)),
            ("more[signature 128x8 h4 ff512 e8]", dict(d_model=128, n_blocks=8, n_heads=4,
                                                       d_ff=512, num_experts=8)),
        ):
            m = MoRE(bundle, name_emb, route_on="signature", k=2, **kw)
            _measure(m, cb, tag, B, rows)
        del cb
        torch.cuda.empty_cache()

    # ---- SetJoin (single-table substrate): gate settings ----
    nn_dict = setjoin_neighbors(bundle, fanout=args.fanout)
    for B in (64, 128):
        loader = make_loader(bundle, task, "train", num_neighbors=nn_dict, batch_size=B,
                             shuffle=False)
        jb = to_join_batch(next(iter(loader)), bundle, ent, wide_len=args.wide_len,
                           set_size=args.set_size).to("cuda")
        wide = int((~jb.wide_is_pad).sum())
        elems = int(jb.elem_mask.sum())
        print(f"SetJoin JoinBatch: B={B} W={args.wide_len} N={args.set_size} "
              f"(real wide {wide/B:.0f}/seed, elems {elems/B:.0f}/seed)")
        for tag, kw in (
            ("setjoin[dense v2]", dict(route_on="dense")),
            ("setjoin[signature e4 v3]", dict(route_on="signature", num_experts=4)),
            ("setjoin[signature e4 +shared]", dict(route_on="signature", num_experts=4,
                                                   use_shared=True)),
            ("setjoin[signature e8]", dict(route_on="signature", num_experts=8)),
            ("setjoin[signature e4 +axial1 v4]", dict(route_on="signature", num_experts=4,
                                                      n_axial_layers=1)),
        ):
            m = SetJoin(bundle, name_emb, ent, d_model=128, n_heads=4, n_wide_layers=2,
                        n_set_layers=2, n_pma=4, k=2, d_sig=64, **kw)
            _measure(m, jb, tag, B, rows)
        del jb
        torch.cuda.empty_cache()

    out = Path("results/complexity")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bench_{gpu.split()[-1].lower()}.json"
    path.write_text(json.dumps(dict(gpu=gpu, dataset=args.dataset, task=args.task,
                                    encoder=args.encoder, iters=args.iters, rows=rows), indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
