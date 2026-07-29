"""probe_met_offset.py — diagnose the MultiEmbeddingTensor offset assert that kills some runs.

`graph.py:_patch_multiembedding_offset` rebases `offset` when `offset[0] != 0`, handling two observed
value layouts (`w == T` and `w == k + T`) and *deliberately* falling through to torch_frame's original
`assert self.offset[0] == 0` for anything else — no silent fix. Array task 6 of the qwen baseline
(rel-f1 / driver-top3) hit a third layout.

This captures the actual `(k, T, w)` on the fall-through path instead of guessing, so the patch can be
extended for a layout we have *seen* rather than one we imagined.

    .venv/bin/python scripts/probe_met_offset.py --dataset rel-f1 --task driver-top3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEEN: list[dict] = []


def install_probe() -> None:
    """Wrap the ALREADY-PATCHED validate and record every case it could not normalise."""
    from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor as _MET

    inner = _MET.validate

    def validate(self):
        off = self.offset
        # Report EVERY offset[0] != 0, including the degenerate numel() < 2 case that the graph.py
        # patch's `numel() >= 2` guard excludes — that exclusion is why the assert still fires.
        if off is not None and off.numel() < 2 and int(off[0]) != 0:
            rec = {"case": "DEGENERATE numel<2", "numel": int(off.numel()),
                   "offset": off.tolist(), "values_shape": tuple(self.values.shape),
                   "n_cols_implied": int(off.numel()) - 1}
            SEEN.append(rec)
            print("\n*** DEGENERATE CASE — zero-column MET with nonzero offset base ***")
            for key, v in rec.items():
                print(f"    {key:20} {v}")
        if off is not None and off.numel() >= 2 and int(off[0]) != 0:
            k = int(off[0])
            T = int(off[-1]) - k
            w = int(self.values.shape[1])
            rec = {
                "k": k, "T": T, "w": w,
                "n_cols": off.numel() - 1,
                "rows": int(self.values.shape[0]),
                "matches_w_eq_T": w == T,
                "matches_w_eq_k_plus_T": w == k + T,
                "offset_head": off[:6].tolist(),
                "offset_tail": off[-4:].tolist(),
                "col_dims": (off[1:] - off[:-1]).tolist()[:8],
            }
            if not (rec["matches_w_eq_T"] or rec["matches_w_eq_k_plus_T"]):
                SEEN.append(rec)
                print("\n*** FALL-THROUGH CASE (the one that asserts) ***")
                for key, v in rec.items():
                    print(f"    {key:24} {v}")
                print(f"    w - T = {w - T}   w - (k+T) = {w - (k + T)}   k = {k}")
        return inner(self)

    _MET.validate = validate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-top3")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="the failing run used 8; the collate/slice then happens IN the worker")
    ap.add_argument("--shuffle", action="store_true",
                    help="training shuffles, which changes the index sets torch_frame slices with")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import torch

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from relbench.tasks import get_task

    install_probe()   # AFTER graph.py's own patch has been applied at import

    bundle = build_gloss_graph(args.dataset)
    task = get_task(args.dataset, args.task, download=True)
    from gloss.utils.seeding import seed_everything

    seed_everything(args.seed)
    loader = make_loader(bundle, task, args.split, batch_size=args.batch_size,
                         num_workers=args.num_workers, shuffle=args.shuffle)

    n_ok = 0
    for i, raw in enumerate(loader):
        if i >= args.batches:
            break
        try:
            to_cell_batch(raw, bundle, task.entity_table, seq_len=args.seq_len, max_fk=5)
            n_ok += 1
        except AssertionError as e:
            print(f"\nASSERT on batch {i}: {type(e).__name__}: {e}")
            break
        except Exception as e:
            print(f"\n{type(e).__name__} on batch {i}: {e}")
            break

    print(f"\n=== {args.dataset}/{args.task} [{args.split}] workers={args.num_workers} "
          f"shuffle={args.shuffle} seed={args.seed}: {n_ok} batches collated OK")
    print(f"=== unrecognised offset layouts seen: {len(SEEN)}")
    if not SEEN:
        print("    (none — this split/task did not reproduce the fall-through)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
