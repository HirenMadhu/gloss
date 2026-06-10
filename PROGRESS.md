# PROGRESS

Phase-by-phase build log for gloss/HALOS (see `implementation.md` for the plan, `idea.md` for rationale).
Append after each phase; stop and report at gates.

---

## Environment (done)
- uv venv, Python 3.12.13. Install via `scripts/setup_env.sh` (staged, custom indexes).
- torch **2.8.0+cu128** (works on the A40 dev node, CUDA 12.8), torch_geometric 2.8.0, relbench 2.1.2,
  pytorch-frame 0.3.0, sentence-transformers 5.5.1, transformers 5.10.2, lightgbm, shap, Lightning 2.6.5.
- **pyg_lib / torch_scatter prebuilt wheels do NOT load on el8 (glibc 2.28 vs required 2.29/2.32)** → not
  installed; we use PyG native ops + our own leakage-safe sampler instead.
- Frozen encoder = `Qwen/Qwen3-Embedding-4B` (2560-dim) — config + tokenizer verified under transformers 5.x.
- flash-attn: source build submitted via `sbatch scripts/build_flash_attn.sh` (sm_86;sm_90). Off the
  Phase 0–2 critical path.

## Phase 0 — leakage-safe data (done, tests green)
- `gloss/data/relbench_graph.py`: `SchemaRegistry` (global ids; distinct `fk_role_id` per FK → dual-FK fix),
  `HeteroTemporalGraph` (PK/FK indices + per-table time arrays over a relbench `Database`),
  `TemporalNeighborSampler` (BFS, **invariant: context cell valid iff `row_time <= seed_time`**),
  `load_task_bundle`.
- `gloss/data/collate.py`: `collate_subgraphs` → padded `[B,T]` `TokenBatch`; pairwise geometry built
  lazily per-segment (no dense `[B,T,T]`).
- `gloss/utils/`: seeding, config (OmegaConf + scratch dir), logging, flops.
- **Bug found & fixed:** `_timestamp_to_ns` returned microseconds for `datetime64[us]` seed dates while
  row times were nanoseconds → no event ever passed the leakage filter and the leakage test passed
  *vacuously*. `test_shapes::neighbors_sampled` caught it; fixed to use `pd.Timestamp(ts).value` (always ns),
  and strengthened `test_leakage` to require timestamped context (non-vacuous).
- **Tests:** `tests/test_leakage.py` (3, incl. real rel-f1), `tests/test_shapes.py` (5) — **8 passed**.
- **DoD:** `python scripts/run_finetune.py --dry-run` prints batch shapes + leakage=0 on synthetic AND
  rel-f1 (9 tables / 67 cols / 14 fk-roles).

## Phase 1 — DocCards + text cache + synthetic — IN PROGRESS
- `gloss/data/synthetic.py`: twin-column planted generator (emits a real relbench `Database`),
  `make_variants` (cross-schema transfer existence proof), `make_synthetic_dualfk` (FK-role test).
- `gloss/data/doccards.py`: `DocCard` + template renderer + regimes (full/name_only/placebo) +
  programmatic synthetic-card renderer (encodes planted sign, no LLM).
- TODO: `text_cache.py` (Qwen embed+cache), `doccard_authoring.py` (Claude-authored real cards + provenance),
  `scripts/build_text_cache.py`, Phase-1 tests.
