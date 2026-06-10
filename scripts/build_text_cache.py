#!/usr/bin/env python
"""build_text_cache.py — Phase 1: render DocCards and embed them once with the frozen Qwen encoder.

    # synthetic (programmatic cards), fast dummy encoder for a smoke test:
    python scripts/build_text_cache.py --dataset synthetic --encoder dummy --regime all
    # real dataset with the real Qwen-4B encoder (GPU; weights cached to scratch):
    HF_HOME=~/scratch60/hf python scripts/build_text_cache.py --dataset rel-f1 --task driver-dnf \
        --encoder qwen --regime all --arm informed
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

REGIMES = ("full", "name_only", "placebo")


def _bundle(args):
    if args.dataset == "synthetic":
        from gloss.data.synthetic import make_synthetic_bundle

        bundle, planted = make_synthetic_bundle(seed=args.seed)
        return bundle, planted
    from gloss.data.relbench_graph import load_task_bundle

    return load_task_bundle(args.dataset, args.task, download=True), None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--task", default="driver-dnf")
    p.add_argument("--encoder", choices=["qwen", "dummy"], default="qwen")
    p.add_argument("--regime", choices=[*REGIMES, "all"], default="all")
    p.add_argument("--arm", choices=["informed", "blind"], default="informed",
                   help="which Claude-authored card set to use for real datasets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    from gloss.data.doccards import cards_for_database
    from gloss.data.text_cache import DummyEncoder, QwenEmbedder, build_text_cache

    bundle, planted = _bundle(args)
    authored = {}
    if args.dataset != "synthetic":
        from gloss.data.doccard_authoring import load_authored_cards

        authored = load_authored_cards(args.dataset, args.arm)
        print(f"[authoring] loaded {len(authored)} Claude-authored cards (arm={args.arm})")

    cards = cards_for_database(bundle.db, bundle.registry, planted=planted, authored=authored)
    encoder = DummyEncoder() if args.encoder == "dummy" else QwenEmbedder()
    regimes = REGIMES if args.regime == "all" else (args.regime,)

    print(f"[cards] {len(cards)} cards over {bundle.registry.num_cols} columns | encoder={args.encoder}")
    for regime in regimes:
        tc = build_text_cache(cards, bundle.registry.num_cols, regime, encoder,
                              dataset=bundle.dataset, force=args.force)
        print(f"   regime={regime:10s} -> emb {tc.emb.shape} (dim {tc.dim})")
    print("TEXT_CACHE_OK")


if __name__ == "__main__":
    main()
