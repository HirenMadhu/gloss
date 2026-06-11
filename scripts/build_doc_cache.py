"""build_doc_cache.py — Phase-1 DoD entry point.

Embeds the prose doc corpus once with the frozen encoder, grounds every schema element, prints a
coverage report, and caches embeddings + the grounding result to disk (idempotent).

    .venv/bin/python scripts/build_doc_cache.py --encoder qwen        # real Qwen3-Embedding-4B (GPU)
    .venv/bin/python scripts/build_doc_cache.py --encoder hash        # cheap wiring smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from gloss.docs.cache import EmbeddingCache, HashEncoder, QwenEncoder  # noqa: E402
from gloss.docs.corpus import DocCorpus, coverage_report, schema_elements_from_db  # noqa: E402
from gloss.docs.grounding import GroundingConfig, ground  # noqa: E402
from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.logging import get_logger  # noqa: E402

log = get_logger("gloss.build_doc_cache")
REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rel-f1")
    ap.add_argument("--dataset", default=None, help="override cfg.data.dataset")
    ap.add_argument("--encoder", choices=["qwen", "hash"], default="qwen")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = load_config(args.config)
    dataset = args.dataset or str(cfg.data.dataset)
    gcfg = GroundingConfig(
        chunk_sentences=int(cfg.docs.grounding.chunk_sentences),
        top_k=int(cfg.docs.grounding.top_k),
        sim_threshold=float(cfg.docs.grounding.sim_threshold),
        temp=float(cfg.docs.grounding.temp),
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO / "data" / "doc_cache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    # corpus + schema elements (from the real relbench schema)
    corpus = DocCorpus.load(REPO / "doc_corpus", dataset)
    from relbench.datasets import get_dataset

    db = get_dataset(dataset, download=False).get_db(upto_test_timestamp=False)
    elements = schema_elements_from_db(db)
    spans = corpus.spans(gcfg.chunk_sentences)
    log.info("dataset=%s | %d schema elements | %d spans | encoder=%s",
             dataset, len(elements), len(spans), args.encoder)

    if args.encoder == "qwen":
        encoder = QwenEncoder(str(cfg.docs.encoder))
    else:
        encoder = HashEncoder(dim=int(cfg.docs.get("d_text", 64)) if args.encoder == "hash" else 64)
    cache = EmbeddingCache(encoder, cache_dir / f"emb_cache_{args.encoder}.pt")

    result = ground(elements, spans, cache, gcfg, regime=args.regime)
    log.info("d_text = %d", result.d_text)

    rep = coverage_report(result.grounded_by_key(), elements)
    print(json.dumps({"dataset": dataset, "encoder": args.encoder, "regime": args.regime,
                      "d_text": result.d_text, "coverage": rep}, indent=2, default=float))

    # persist the grounding result (gather-by-key at train time)
    out = cache_dir / f"grounding_{args.encoder}_{args.regime}.pt"
    torch.save(
        {"keys": result.keys, "emb": result.emb, "rel": result.rel,
         "grounded": result.grounded, "d_text": result.d_text}, out,
    )
    log.info("saved grounding -> %s ; emb cache -> %s", out, cache.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
