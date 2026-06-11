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
