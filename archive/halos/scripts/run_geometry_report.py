"""run_geometry_report.py — Phase-3 DoD: render the compiled per-FK-role kernels.

Builds rel-f1, an (untrained) HALOS model, compiles the geometry from the cached doc grounding, and
prints the per-metapath Gaussian-in-τ kernels. Pipeline check — no training needed.

    .venv/bin/python scripts/run_geometry_report.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from gloss.docs.cache import EmbeddingCache, HashEncoder  # noqa: E402
from gloss.docs.corpus import DocCorpus, schema_elements_from_db  # noqa: E402
from gloss.docs.grounding import GroundingConfig, ground  # noqa: E402
from gloss.eval.geometry_report import format_report, geometry_report  # noqa: E402
from gloss.model.halos import HALOS, build_doc_per_metapath  # noqa: E402
from gloss.utils.config import load_config  # noqa: E402
from gloss.utils.seeding import seed_everything  # noqa: E402


def main() -> int:
    warnings.filterwarnings("ignore")
    cfg = load_config("rel-f1")
    seed_everything(int(cfg.seed))

    from gloss.data.graph import build_gloss_graph
    from relbench.datasets import get_dataset

    bundle = build_gloss_graph(str(cfg.data.dataset))
    db = get_dataset(str(cfg.data.dataset), download=False).get_db(upto_test_timestamp=False)

    # ground FK-role docs (hash encoder keeps this offline/fast; swap to a real qwen cache for the figure)
    corpus = DocCorpus.load(Path(__file__).resolve().parents[1] / "doc_corpus", str(cfg.data.dataset))
    elements = schema_elements_from_db(db)
    d_text = 64
    enc = EmbeddingCache(HashEncoder(dim=d_text))
    g = ground(elements, corpus.spans(3), enc, GroundingConfig(sim_threshold=0.0), regime="full")
    doc_per_mp = build_doc_per_metapath(bundle, g)

    model = HALOS(bundle, d_model=64, n_heads=int(cfg.model.n_heads), n_layers=2, d_text=d_text, n_freq=8)
    rep = geometry_report(model, bundle, doc_per_metapath=doc_per_mp)
    print(format_report(rep, head=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
