# PROGRESS — DOC-RT

Running log of the DOC-RT build (RT cell-token substrate + per-column documentation via FiLM). The retired
HALOS log is at `archive/halos/PROGRESS_halos.md`. Design: `CLAUDE.md`, `idea.md`, `implementation.md`.

## 2026-06-24 — Pivot HALOS → DOC-RT; P0–P4 scaffold green; headline gate launched

**Pivot.** Retired the HALOS geometry stack (Gaussian-in-τ kernels, `g_θ` geometry generator, dimensionless
time `τ`, scale-equivariance) to `archive/halos/`. DOC-RT keeps RT's cell-token substrate + relational masks
and adds one signal: per-column documentation injected by FiLM into the cell encoder. `gloss` package kept.

**Phase 0 — substrate [done].** `data/graph.py` (RelBench → hetero temporal graph, `fk_role_id`, leakage-safe
sampler) + `data/collate.py` (`CellBatch`: RT cell tokens at fixed `seq_len`, the 4 relational masks rebuilt in
the model, `f2p_nbr_idxs`, `is_seed_cell`, `cell_placement`). DoD: `run_train.py --dry-run` builds a real
rel-f1 cell batch (B=8, 527 real cells across all 5 tables), leakage check = 0, forward OK.

**Phase 1 — grounding [done].** `docs/{corpus,grounding,cache}.py`; regimes `full | null | shuffled | name_only`;
`d_null` fallback; Qwen3-Embedding-4B grounding cached at `data/doc_cache/rel-f1/` (complete — no LM forward at
train time).

**Phase 2 — cell encoder [done].** `model/column_encoder.py` `CellEncoder`: pytorch-frame dtype encoders +
FiLM by `d_c` + RT name token + learned `d_null`; scatters per-cell vectors into the `[B,S,d]` grid.

**Phase 3 — model [done].** `model/rt_substrate.py` (RT `RelationalBlock`: col/feat/nbr/full SDPA attention +
SwiGLU, pre-norm RMSNorm) + `model/docrt.py` (`DOCRT` = CellEncoder → RTSubstrate → EntityHead).

**Phase 4 — training + HEADLINE GATE [running].** `train/*`, `eval/ablation.py`, `scripts/run_headline.{py,sh}`.
SLURM job **28967996**: array `0-19%4` = 5 seeds × {full, null, shuffled, name_only}, h100:1, 10 epochs, qwen.
Aggregate when done: `run_headline.py --aggregate` → `results/headline/`. Reading: `full > null` (signal)
`> shuffled` (meaning) `> name_only` (beats names); `full ≈ null` is the honest negative.

**Tests: 56 passed, 1 skipped (flash-attn).** Leakage, grounding (null fallback / placebo / name-independence /
determinism), FiLM-responds, FK-role distinctness, cell-batch shapes / collate / column-encoder, single-batch
overfit + grad-flow (`test_train.py`), gate bookkeeping (`test_ablation.py`).

**Engineering decisions.**
- Memory: DOC-RT attention is dense `O(S²)` with 4 `[B,S,S]` bool masks/block; SDPA+bool-mask → math backend
  materializes all score matrices for backward. Measured rel-f1 cells/seed (mean 206, max 353) → `seq_len=512`
  (0% truncation). Gate `batch_size=64` (the archived HALOS value 512 was infeasible for cell-token attention).
- `run_train.py` flags added for fast smokes: `--seq-len`, `--limit-val-batches`, `--num-workers`.

**Deferred (P5–8).** P0b self-labels (task-table node type + seed's past task rows, leakage-safe); coverage-vs-
gain curve; more DBs; same-column attention-bias ablation (`--docbias`); masked-cell pretraining + transfer;
the synthetic twin-column DB + audit (existence proof when docs are the only disambiguator).
