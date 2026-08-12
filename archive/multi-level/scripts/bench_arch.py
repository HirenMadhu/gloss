#!/usr/bin/env python
"""Matched per-step wall-clock: single-level (arch=rt) vs two-level, same config, same batch.

WHY NOT just read SLURM job times: total job wall-clock conflates per-step compute with how many
epochs early stopping ran. In the completed grid, driver-top3 took 2.0 min under one lr and 11.4 min
under another at the SAME architecture — that spread is convergence, not speed. To compare models you
have to fix the batch and time the step.

Reports forward and forward+backward ms/step plus parameter counts. Run on GPU for numbers that
transfer to training; on CPU the ratio is still informative but the absolute values are not.

    python scripts/bench_arch.py --seq-len 512 --batch-size 8 --iters 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _time(fn, iters, warmup, sync):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import torch

    from gloss.model.more import MoRE
    from gloss.text.schema import build_table_name_embeddings, role_name_embeddings_with_none
    sys.path.insert(0, str(REPO / "tests"))
    from tests._relf1 import name_table, sample_cell_batch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    print(f"# device={dev.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else " — ratios transfer, absolute ms do NOT")
          + f" | rel-f1 | seq_len={args.seq_len} batch={args.batch_size} "
            f"d_model={args.d_model} n_blocks={args.n_blocks}")

    bundle, _task, cb = sample_cell_batch(seq_len=args.seq_len, batch_size=args.batch_size)
    cb = cb.to(dev)
    name_emb = name_table().to(dev)

    class _Hash:
        """Cheap stand-in for the frozen encoder — the benchmark must not measure text encoding."""

        def __call__(self, texts, kind="query"):
            from gloss.text.cache import HashEncoder

            return HashEncoder(dim=64)(texts, kind)

        def encode(self, texts, kind="query"):
            return self(texts, kind)

    common = dict(d_model=args.d_model, d_sig=64, n_blocks=args.n_blocks, n_heads=8,
                  d_ff=args.d_model * 4, enc_channels=args.d_model,
                  route_on="signature", num_experts=4, k=2)
    enc = _Hash()
    two_level_kw = dict(
        arch="two_level",
        table_name_emb=build_table_name_embeddings(bundle, enc, kind="query").to(dev),
        role_name_emb=role_name_embeddings_with_none(bundle, enc, kind="query").to(dev),
    )

    rows = []
    for label, extra in (("single-level (rt)", {}), ("two-level", two_level_kw)):
        torch.manual_seed(0)
        model = MoRE(bundle, name_emb, **common, **extra).to(dev)
        n_par = sum(p.numel() for p in model.parameters())

        def fwd():
            with torch.no_grad():
                model(cb)

        def fwd_bwd():
            model.zero_grad(set_to_none=True)
            logits, aux = model(cb)
            (logits.squeeze(-1).sum() + aux).backward()

        model.eval()
        f = _time(fwd, args.iters, args.warmup, sync)
        model.train()
        fb = _time(fwd_bwd, args.iters, args.warmup, sync)
        rows.append((label, n_par, f, fb))
        print(f"{label:>20}  params={n_par/1e6:6.2f}M  fwd={f:8.1f} ms  fwd+bwd={fb:8.1f} ms", flush=True)

    (_, p0, f0, fb0), (_, p1, f1, fb1) = rows
    print(f"\n{'two-level / single-level':>28}: params {p1/p0:.2f}x   "
          f"fwd {f1/f0:.2f}x   fwd+bwd {fb1/fb0:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
