# SetJoin — build spec (branch `setjoin`)

*Contracts (shapes, semantics) are normative; see [setjoin_idea.md](setjoin_idea.md) for the why.
Everything lives in `gloss/setjoin/` — no file under `gloss/{model,data,train,eval}` is modified.*

## Reuse boundary

Reused unchanged: `build_gloss_graph`/`GraphBundle`/`make_loader` (temporal as-of sampler),
`coerce_binary_target`, `canonical_relation`/`is_forward_relation`/`FK_NONE`/`relation_fk_role`,
`column_vocab`/`feature_col_names`, `CellEncoder.encode_type` (NEVER its `forward`),
`gloss/text/` + `name_embeddings`, `task_loss`, `binary_metrics`/`regression_metrics`,
`task_kind`/`target_stats`/`_BestValState`, `MoREDataModule`, and `eval/ablation.py`'s
`dataset_tasks`/`build_grid`/`aggregate`/`format_table` bookkeeping.

Replaced: `to_cell_batch`/`CellBatch` → `to_join_batch`/`JoinBatch`; `MoRE` → `SetJoin`;
thin forks of `MoRELitModule`, `test_eval.predict_split/evaluate_split`, `ablation.run_config`.

## Package layout

```
gloss/setjoin/
    paths.py       # schema maps: parent_rels / child_rels / m2o_paths / setjoin_neighbors
    recency.py     # fixed log-spaced Δt buckets (standalone; bin 0 = untimed/pad)
    collate.py     # JoinBatch + to_join_batch
    model.py       # SetJoin (WideEncoder, RowPool, SetEncoder+PMA, head)
    train.py       # SetJoinLitModule + train_prebuilt_setjoin
    eval.py        # predict_split / evaluate_split (JoinBatch-native)
    runner.py      # run_config (SLURM array cell) + compare_table
scripts/run_setjoin.py   # --dry-run [--collate-only] | --train | --list | --index | --aggregate | --compare
scripts/run_setjoin.sh   # Milgram-native SLURM array runner (gpu / h100:1, %8)
tests/_join_fixtures.py  # 3-level chain fixture (payment→order→customer→region)
tests/test_join_collate.py  test_join_leakage.py  test_setjoin_model.py
tests/test_setjoin_train.py test_setjoin_runner.py
```

## Sampling contract

Seeds and as-of correctness come from `make_loader` (`NeighborSampler(time_attr="time",
temporal_strategy="last", disjoint=True)`). Fanout is a **per-edge-type dict** from
`setjoin_neighbors(bundle, fanout=64)`. **PyG direction semantics (the v1 gate had this inverted —
every seed got ~1 child per relation and elements lost their flattened parents):** `num_neighbors[et]`
budgets how many *src*-side neighbors are drawn per frontier node of type `et.dst`, so:

- forward `f2p_*` types (`(child, f2p_col, parent)`, dst=parent) pull **children** → `[fanout, 0]` —
  up to `fanout` most-recent children per relation at hop 1; **no** hop-2 o2m expansion (no
  children-of-children, no siblings of parents).
- reverse `rev_f2p_*` types (dst=child) pull the unique **parent** → `[1, 1]` — the seed's parents at
  hop 1; grandparents and each child row's own flattened parents at hop 2.

This samples exactly the m2o closure + 1-hop children. Guarded by a rel-f1 regression test
(`test_sampler_multiplicity_and_parent_flattening_rel_f1`) asserting real child multiplicity and
parent flattening on sampled data — synthetic-batch tests bypass the sampler and cannot catch a
direction inversion. If relbench's `NeighborSampler` rejects the dict, fall back to a flat list
`[fanout, 8]` (collate ignores the extra rows; enable tf_dict row subsetting for encode cost).

**PyG facts the collate must respect** (stress-tested):
- Sampled edges are stored under the edge type *traversed*: children arrive via `rev_f2p_*` types.
  Adjacency must be built from ALL edge types, canonicalized by direction, deduped.
- `disjoint=True` samples a **tree**: the same DB row can appear as several node copies. Seed
  self-exclusion and ancestor dedup compare **global `n_id`**, never local indices.
- The entity-type store spans seeds + sampled entity-type neighbors; seeds are rows
  `[:seed_time.numel()]`; `input_id` is per-seed `[B]`.

## `JoinBatch` (the collate contract)

Per seed: a cell-granular **wide row** `[B, W]` and a row-granular **union set** `[B, N]`.

| field | shape | dtype | semantics |
|---|---|---|---|
| `wide_col_idxs` | [B,W] | long | global (table,column) id via `column_vocab`; −1 pad/marker |
| `wide_table_idxs` | [B,W] | long | node-type id; −1 pad |
| `wide_path_idxs` | [B,W] | long | m2o join-path id (0 = seed's own row); −1 pad |
| `wide_is_pad` / `wide_missing` | [B,W] | bool | pad slot / absent-parent marker slot |
| `wide_row_time` / `wide_is_timed` | [B,W] | f64/bool | for Δt recency |
| `elem_mask` | [B,N] | bool | True = real element |
| `elem_rel_idxs` | [B,N] | long | `relation_fk_role` of the child→seed relation; 0 pad |
| `elem_table_idxs` | [B,N] | long | child node-type id; 0 pad |
| `elem_row_time` / `elem_is_timed` | [B,N] | f64/bool | child-row time |
| `elem_hop` | [B,N] | long | 1 everywhere (MVP; reserved) |
| `child_counts` | [B,R] | f32 | per child-relation counts, post-sampler pre-truncation |
| `seed_time`/`target`/`has_target`/`input_id` | [B] | f64/f32/bool/long | `input_id` −1 if absent |
| `tf_dict` | — | — | node_type → TensorFrame (as sampled) |
| `wide_placement` | — | — | nt → `(b, w, row_in_tf, col_pos)` cell scatter into [B,W,d] |
| `elem_rows` | — | — | nt → `(b, n, row_in_tf, path_id)` additive row scatter into [B,N,d] |

`elem_rows.path_id`: 0 (=`FK_NONE`) for the child row itself; `relation_fk_role(prel)` for each of the
child's flattened parents. `.to(device)` and `.pretty_shapes()` mirror `CellBatch` (pretty_shapes also
reports wide/set truncation and empty-set counts).

### `to_join_batch` semantics

1. Wide row emits paths in `m2o_paths` order, seed first (survives truncation, cap `wide_len`);
   a walk that dies (NULL FK / sampler miss / temporal exclusion) emits exactly ONE marker slot
   (`wide_missing=True`) at that path. Featureless tables emit nothing. An ancestor row already
   emitted (by n_id) is not emitted twice.
2. Union set: children of the seed across all child relations, one element per (child row × arriving
   relation), sorted most-recent-first (untimed last, stable), truncated to `set_size`. Dual-relation
   children appear once per relation with distinct `elem_rel_idxs` — intended. Inside an element, the
   seed itself is excluded from the flattened parents (by n_id); a missing non-seed parent contributes
   nothing.
3. No cross products anywhere: Σ`elem_mask` == Σ`child_counts` whenever under the cap.
4. Leakage: no wide or elem `row_time > seed_time` (timed rows) — guaranteed by the sampler, asserted
   in tests.

## `SetJoin` model

`SetJoin(bundle, name_emb, *, d_model=128, enc_channels=None, n_wide_layers=2, n_set_layers=2,
n_heads=4, n_pma=4, dropout=0.1, out_dim=1)`; `forward(jb) -> (logits [B,out_dim], aux)` with
`aux = 0` (keeps MoRE's `(logits, aux)` contract so the training forks stay thin).

- Cells: `CellEncoder.encode_type(nt, tf)` → `[n, C, d_model]` (frozen name token + stype value enc).
- `WideEncoder`: scatter wide cells → `[B,W,d]`; + `path_emb + recency_emb`; missing markers =
  `missing_emb + path_emb`; CLS prepended (never masked); `n_wide_layers` pre-norm transformer layers
  with `src_key_padding_mask = wide_is_pad`; CLS → `seed_repr [B,d]`.
- `RowPool`: one shared gated-attention pool over a row's C cells → row vector (name tokens already
  disambiguate columns/tables).
- Element assembly: `E[B,N,d]` = additive `index_put_(accumulate=True)` of
  `RowPool(rows) + fk_emb(path_id)` per `elem_rows`, then + `fk_emb(elem_rel_idxs) +
  table_emb(elem_table_idxs) + recency_emb(Δt bin) + hop_emb(elem_hop)`; LayerNorm. `table_emb` is
  load-bearing: `fk_role_id` is keyed by canonical FK column NAME and collides across tables.
- `SetEncoder`: concat learned `null_elem` (always unmasked — empty sets stay well-defined);
  `n_set_layers` transformer layers (key_padding_mask from `elem_mask`; NO positional encoding —
  permutation invariance is a tested contract); seed-conditioned PMA readout: queries
  `pma_emb + W_q(seed_repr)`, cross-attention → flatten → `context [B,d]`.
- Head: `MLP(LayerNorm([seed_repr ; context ; W_cnt(log1p(child_counts))])) → [B, out_dim]`;
  `out_dim=1` for binary and regression (regression stays z-scored in the Lit module).

## Training / eval / runner

- `SetJoinLitModule` = `MoRELitModule` fork: `SetJoin` instead of `MoRE`, `to_join_batch` in
  `transfer_batch_to_device`, no aux term in the loss. Identical `val/*` metric names (best-val
  selection and early stopping monitor `val/auroc` max / `val/mae` min), identical regression
  standardization, nan-grad guard kept.
- `train_prebuilt_setjoin` mirrors `finetune.train_prebuilt` (same trainer flags,
  `gradient_clip_val=1.0`, patience 3, `seed_everything` first) with
  `num_neighbors=setjoin_neighbors(bundle, fanout)`.
- `eval.predict_split/evaluate_split` mirror `test_eval` (per-seed `input_id` alignment,
  de-standardization, `task.evaluate` on the coerced target table).
- `runner.run_config`: grid = `build_grid(dataset_tasks(("rel-f1","rel-trial","rel-event"),
  ["leaderboard"]), seeds=3, signals=("setjoin",))` → **27 cells**; record schema matches the ablation
  JSONs plus **`test_nmae` stored at run time** (leaderboard regression metric); OOM-retry à la
  run_gridsearch. `compare_table` prints SetJoin vs RT-from-scratch / GelGT
  (`results/leaderboard_baselines.json`) / MoRE grid best.

## Experiment plan (the gate)

Defaults: d_model 128, heads 4, 2 wide + 2 set layers, n_pma 4, wide_len 128, set_size 128, fanout 64,
batch 128, lr 3e-4, wd 0.01, ≤30 epochs + early stop, seeds 0–2, encoder harrier (caches exist for all
3 DBs). Verify rel-event scratch symlinks first, then:

```bash
N=$(.venv/bin/python scripts/run_setjoin.py --list)          # 27
sbatch --array=0-$((N-1))%8 scripts/run_setjoin.sh --out-dir results/setjoin_v1
.venv/bin/python scripts/run_setjoin.py --aggregate --out-dir results/setjoin_v1
.venv/bin/python scripts/run_setjoin.py --compare  --out-dir results/setjoin_v1
```

**Stop and report at the gate.** Honest negatives welcome (see falsifiers in the idea doc).

## Test matrix (added invariants)

- `test_join_collate.py` (hermetic): m2o path walk (depth-2, seed-first, marker, n_id ancestor dedup);
  union built from rev-only edge instances; dual-FK distinct `elem_rel_idxs`; seed n_id exclusion
  inside elements; Δt-sort truncation; `child_counts`; no-cross-product; shapes/dtypes; `input_id=-1`.
- `test_join_leakage.py` (hermetic + rel-f1-guarded): no `row_time > seed_time` in wide or set;
  planted future row detected; real-loader spot check.
- `test_setjoin_model.py`: hermetic sub-module tests on raw tensors (SetEncoder permutation
  invariance, empty-set null_elem grads, missing-marker path, RowPool); rel-f1-guarded full-model
  forward/grad-flow/param-count and end-to-end permutation check.
- `test_setjoin_train.py`: overfit-one-batch binary + regression (guard follows test_train.py's
  convention); de-standardization round-trip.
- `test_setjoin_runner.py`: grid == 27; record → aggregate/format_table round-trip; NMAE math;
  compare-table rendering.

Phases S0–S4 with DoD commands live in the approved build plan; each phase ends green on
`.venv/bin/python -m pytest tests/` + its DoD, appends `PROGRESS.md`, commits `feat(setjoin-N): …`.
