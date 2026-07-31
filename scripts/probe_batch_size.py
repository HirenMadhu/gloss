#!/usr/bin/env python
"""Find the largest feasible training batch for a two-level config, per dataset.

GelGT trains at batch 512; we train at 64. Before adopting their batch we have to know what fits:
our attention is dense ``O(S^2)`` with four boolean masks per block (SDPA falls back to the math
backend, which materialises the score matrices for backward), and the MoE dense-combine MVP runs
ALL experts on ALL tokens. So memory does not scale the way a plain transformer's would, and the
answer has to be measured rather than assumed.

Runs a real 2-step fit at each batch size and reports peak CUDA memory, stopping at the first OOM.
Uses `train_prebuilt` so the measured path is the one training actually takes.

    python scripts/probe_batch_size.py --dataset rel-f1 --task driver-position
    python scripts/probe_batch_size.py --dataset rel-trial --task site-success --d-model 256
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-position")
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-fk", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="*", default=[64, 128, 256, 512])
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import torch
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none
    from gloss.train.finetune import _name_encoder, name_embeddings, train_prebuilt
    sys.path.insert(0, str(REPO / "scripts"))
    from run_ablation_phases import TWO_LEVEL_PHASES

    if not torch.cuda.is_available():
        print("no CUDA — this probe is meaningless on CPU")
        return 1
    d_text = 5376 if args.encoder == "harrier" else 2560
    bundle = build_gloss_graph(args.dataset, cache_dir=str(REPO / "data" / "graph_cache" / args.dataset))
    task = get_task(args.dataset, args.task, download=False)
    name_emb = name_embeddings(bundle, args.dataset, encoder=args.encoder, d_text=d_text)
    enc = _name_encoder(args.dataset, encoder=args.encoder, d_text=d_text)
    mk = dict(d_model=args.d_model, n_blocks=args.n_blocks, n_heads=8, d_ff=args.d_model * 4,
              num_experts=4, enc_channels=args.d_model, k=2, arch="two_level",
              table_name_emb=build_table_name_embeddings(bundle, enc, kind="query"),
              role_name_emb=role_name_embeddings_with_none(bundle, enc, kind="query"),
              **TWO_LEVEL_PHASES["full"])

    dev = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"# {dev}  {total:.1f} GiB | {args.dataset}/{args.task} "
          f"d_model={args.d_model} n_blocks={args.n_blocks} seq_len={args.seq_len}")
    print(f"{'batch':>7} {'peak GiB':>10} {'result':>10}")
    ok = None
    for bs in args.sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            train_prebuilt(bundle, task, name_emb, model_kwargs=dict(mk), route_on="signature",
                           lambda_ortho=0.5, seq_len=args.seq_len, max_fk=args.max_fk,
                           batch_size=bs, max_epochs=1, seed=0, num_workers=args.num_workers,
                           limit_train_batches=2, limit_val_batches=1, early_stop=False)
            peak = torch.cuda.max_memory_allocated() / 2**30
            ok = bs
            print(f"{bs:>7} {peak:>10.2f} {'OK':>10}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>7} {'--':>10} {'OOM':>10}", flush=True)
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{bs:>7} {'--':>10} {'OOM':>10}", flush=True)
            break
    print(f"\nMAX_FEASIBLE_BATCH={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
