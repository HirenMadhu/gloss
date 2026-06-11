# PROGRESS

HALOS / `gloss` — v3 (documentation-conditioned geometry). Build log; one section per phase.

## Bootstrap (env + restructure)
- Spec rewritten v2→v3 (method paper; doc-generated attention geometry is the core). Rewrote `CLAUDE.md`
  to v3; updated project memory.
- Env was already provisioned (`.venv`, py3.12, torch 2.8.0+cu128, PyG 2.8, torch_frame 0.3, relbench,
  sentence-transformers, transformers, lightgbm, shap, lightning, hydra, wandb). Added build deps
  (`wheel ninja einops`) and the PyG compiled extensions (`pyg-lib`, `torch-scatter`, `torch-sparse`
  for `torch-2.8.0+cu128`) — the relbench temporal sampler needs `pyg-lib`.
- `tests/test_env.py`: imports + cuda12/cxx11abi asserts green; flash-attn xfail until built.
- flash-attn: prior build failed (venv lacked `wheel`). Fixed deps + resubmitted `scripts/build_flash_attn.sh`
  (SLURM job; archs 86;90). Its only consumer is the frozen Qwen encoder — off the Phase 0-3 critical path.
- Deleted the stale v2 scaffold (`proxy/ext/audit/train`, v2 model/eval, structured-card tests). New v3
  tree under `gloss/{data,docs,model,eval,utils}`; configs `default.yaml` + `rel-f1.yaml`.

## Phase 0 — substrate ✅
- `gloss/data/graph.py`: `build_gloss_graph` (wraps `make_pkey_fkey_graph` + `get_stype_proposal`,
  cheap deterministic `HashTextEmbedder` for cell text), per-DB vocabs (`node_type_id`; `fk_role_id` /
  `metapath_id` **canonicalized to the fkey column** so forward/reverse share a role and dual FKs stay
  distinct); `make_loader` (per-seed **disjoint** leakage-safe temporal `NeighborSampler`, `time_attr=time`,
  `temporal_strategy=last`).
- `gloss/data/collate.py`: `to_gloss_batch` → dense `GlossBatch` (node-level `[B,N_max]`; pairwise
  `[B,N_max,N_max]` attend_mask / metapath_id / fk_role_id / dt / tau / temporal_valid; per-seed
  seed_time + T_ctx). `tau = log(dt/T_ctx)` on both-timed attendable pairs (float64); timeless / zero-gap /
  >1-hop / pad → structural bucket (temporal_valid=False). Carries per-type `TensorFrame`s + `placement`
  for Phase-2 encoding.
- `gloss/utils/{seeding,config,logging}.py`.
- `scripts/run_finetune.py --dry-run`: builds rel-f1, samples a disjoint batch, prints shapes, leakage check = 0.
- **Tests (hermetic; real rel-f1 guarded by cache):** test_env, test_graph, test_collate, test_leakage,
  test_shapes — **36 passed, 1 xfailed**.
- **Decisions / deferrals:** self-label task-table nodes deferred to Phase 4 (relbench's graph excludes the
  task table; self-labels are a transfer-phase concern). Attention is dense within a seed's subgraph over
  ≤2-hop-reachable pairs (1-hop = FK relation, 2-hop = MULTIHOP bucket, else masked).
- DoD met: `run_finetune.py --dry-run` prints a rel-f1 batch; leakage/shape tests green.

## Phase 1 — doc corpus + grounding ✅
- `doc_corpus/rel-f1/docs.md` + `meta.yaml`: **Tier-2, blind** senior-dev prose (authored from schema +
  F1 domain knowledge only; no task/label/target referenced; ~65% coverage target; `constructor_results`
  + some columns deliberately undocumented). meta carries the blind attestation for the audit.
- `gloss/docs/corpus.py`: load/validate (blind ⇒ attestation required), sentence-chunk into spans,
  enumerate schema elements (table / column / fk_role) from a relbench DB or a spec dict, coverage report.
- `gloss/docs/grounding.py`: chunk→embed→top-K cosine→softmax-pool → `d_e`, `rel_e`; `d_null` fallback
  below threshold; regimes `full | null | shuffled_spans`. **Placebo fix:** permuting span rows is a no-op
  (sims recomputed), so `shuffled_spans` permutes the *element→doc assignment* (derangement) — same
  coverage/length, decorrelated meaning.
- `gloss/docs/cache.py`: `QwenEncoder` (frozen Qwen3-Embedding-4B via sentence-transformers, instruction
  on queries only), `HashEncoder` (dev/tests), idempotent content-hash `EmbeddingCache`.
- `scripts/build_doc_cache.py`: real Qwen run → **d_text=2560 confirmed**, idempotent (cache reload, no
  model reload), coverage emitted.
- **Calibration finding:** Qwen sims sit ~0.52–0.85 (high baseline), so the spec's 0.3 threshold grounds
  everything (100%). Calibrated `sim_threshold=0.60` → **0.725 partial coverage** (table .78 / col .67 /
  fk .92). Encoder-specific knob; revisit at the H1 gate (Phase 5).
- **Tests:** test_corpus, test_grounding (controllable BoW encoder + HashEncoder; null fallback, placebo
  decorrelation, determinism, cache idempotency) — all green. Full suite: **48 passed, 1 xfailed**.

## Phase 2 — doc-conditioned node encoder ✅
- `gloss/model/time_encoding.py`: `node_tau` (τ_u = log((seed−row)/T_ctx), float64, valid only for
  timed positive-gap rows; **scale-invariant** by construction) + `BochnerTime` (learnable Fourier
  features of τ).
- `gloss/model/column_encoder.py`: per-cell embeddings via pytorch-frame `StypeWiseFeatureEncoder`
  (reused, not reinvented) → **per-column FiLM** `x_c=γ(d_c)⊙W_v Enc(v_c)+β(d_c)` → **doc-keyed
  attention pool** over columns → `h_u = pool + E_type(t) + W_t φ(τ_u)`. Ungrounded columns / null
  regime fall back to a learned `d_null`.
- **Tests:** test_time_encoding (formula, scale-invariance, Bochner) hermetic; test_column_encoder
  (guarded rel-f1): finite `[B,N,d_model]`, pad=0, and **FiLM responds to full↔null regime**.
- DoD met. Full suite: **53 passed, 1 xfailed**.

## Phase 3 — the core operator (doc-generated geometry) ✅
- `gloss/model/bias_generator.py` (**CORE**): `g_θ` maps `ctx(p)=[E_metapath(p); doc(p)]` → per-head
  `(a,μ,σ,b)` (σ=softplus+floor). `compile()` runs once per DB over the metapath set → `GeometryTable`
  (recomputed each forward; gradients flow). `absolute_anchor` emits an extra `log T_ctx` coefficient.
- `gloss/model/attention.py`: `B_h(i,j)=b + temporal_valid·a·exp(−(τ−μ)²/2σ²)`; additive bias via
  `F.scaled_dot_product_attention` (SDPA mem-efficient; flash-attn can't take per-pair bias). Diagonal
  kept finite so padded rows don't NaN.
- `gloss/model/halos.py`: `ColumnEncoder` → compile geometry → `HALOSLayer` (pre-norm attn + FFN) stack →
  `EntityHead` seed readout. `build_doc_per_metapath` pools FK-role docs into the geometry context.
- `gloss/model/heads.py`, `gloss/eval/geometry_report.py` (+ `scripts/run_geometry_report.py`): the
  readable per-FK-role kernel exhibit (renders untrained).
- **Invariance fix:** absolute timestamp *cell features* are dropped (time enters only via τ); and
  `T_ctx` now folds in node recencies `(seed−row)` so it always scales with the clock (the old 1.0
  fallback made the node-time term non-invariant). Result: **logits bit-identical under t→c·t (1e-5)**.
- **Tests:** test_bias_generator, test_attention, test_fk_role (dual-FK fixture: distinct geometry +
  role-swap changes preds), test_scale_equivariance (full rel-f1 model invariant; absolute_anchor breaks
  it). DoD: forward runs, geometry report renders. **65 passed, 1 xfailed.**
