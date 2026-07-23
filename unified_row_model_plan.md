# PLAN: Unified hierarchical row model (one architecture, no wide/set split)

**Branch:** `setjoin`  ·  **Status:** DRAFT — confirm the OPEN DECISIONS (§1) before execution.

This retires the two-stream SetJoin model (separate `WideEncoder` over the wide m2o row + `SetEncoder`
over a pre-pooled o2m union set) and replaces it with **one hierarchical model over a set of wide rows**:

```
cell encoder (cell-level MHSA + MoE)   →   learns cell embeddings
      ↓
slot-attention RowPool                  →   one embedding per wide row
      ↓
row encoder (row-level MHSA + MoE)      →   contextualized wide-row embeddings
      ↓
aggregate (mean, or slots = #labels)    →   one prediction-level embedding
      ↓
prediction head                         →   logits
```

The mechanism (MoRE's signature-routed `MoEFFN`, `gloss/model/moe.py`) is used **everywhere** — in the
cell encoder and in the row encoder. There is now exactly one encoder pattern, applied at two granularities
(cells within a row; rows within a seed).

---

## 0. The unit: a denormalized wide row (parent columns repeated — the "one big table")

Denormalization is a standard left-join into **one big table (OBT)**: a row is joined with everything it
points to (its **many-to-one** parents), and the **one-side columns are repeated** across every child.

Concretely, a parent row `x, y, z` (the target `t` is an *external label*, never a feature) with children
`(a1, b1)` and `(a2, b2)` denormalizes to:

```
row 1:   x, y, z, a1, b1        (parent cols repeated)
row 2:   x, y, z, a2, b2        (parent cols repeated)
```

So **per seed the input is a SET of denormalized wide rows** — one per direct child — where each row
carries **(i)** the child's own cells, **plus (ii)** the child's full m2o closure repeated in, **including
the seed and the seed's own parents**. Every row is thus a self-contained denormalized record. **Row 0** is
the seed's own wide row (just its m2o closure), so a childless seed still has ≥1 row. Because table schemas
differ, "columns" are not fixed slots: each cell is a **token tagged by its (column-name, table) identity**
(the frozen-LM name embedding), so heterogeneous wide rows coexist in one tagged-cell grid.

**Scope for now:** o2m is **1-hop only** — the seed's *direct* children. The nested / hop-2 o2m join
(children-of-children) is **deferred**: it fans out multiplicatively and cannot be materialized. m2o
(parent) denormalization is exact and repeated freely at any depth, because it never blows up — this is
where the seed's parent context enters every row.

**Collate change vs today:** today's `elem_rows` scatter *excludes* the seed from each child's flattened
parents ([collate.py:385](gloss/setjoin/collate.py#L385), "never re-inject the seed"). This design does the
**opposite** — the seed (and its m2o closure) is **repeated into every child row** (the `x, y, z` above).
That is the one behavioral change to the join; everything upstream (sampler, temporal cutoff) is unchanged.

---

## 1. OPEN DECISIONS — confirm before we cut code

1. **Row set membership.** `{seed wide row} ∪ {each direct (hop-1) child wide row}`, capped at `M_rows`
   (default 128), most-recent-first. o2m is **1-hop only** (nested/hop-2 deferred, §0). Each child row is
   m2o-denormalized with the **seed + its parents repeated in** (§0 — the one collate change). ← confirm.
2. **Aggregate over rows.** Assumed a **config arm** `aggregate ∈ {mean, slot}`; default `mean`. For `slot`,
   `n_slots = out_dim` (= 1 for every current binary/regression task — so it's a single learned-query pool,
   ≈ PMA-with-1-query, *not* count-blind mean). ← confirm default `mean`.
3. **RowPool (cells→row).** Assumed **slot-attention pool** with `n_cell_slots` (default 1 stage, `C→1`;
   the `C→k1→k2→1` cascade is available but off — narrow rows here, ~≤32 cells). ← confirm 1-stage default.
4. **Keep the raw `child_counts` head feature?** Assumed **dropped** (aggregation is over representations;
   you argued count-awareness is second-order there). Cheap to keep as a config flag if you want it. ← confirm.
5. **Naming.** New model class `RowModel` in a new file `gloss/setjoin/row_model.py`; the existing
   `SetJoin` in `model.py` is **left intact** (still runnable for comparison) until the gate says otherwise.
   ← confirm you want the old model kept for A/B, not deleted.

---

## 2. Shapes (standing config: `d=256`, `M_rows=128`, `C=48`, `d_sig=64`, MoE `M=4/k=2`)

| symbol | meaning | value |
|---|---|---|
| `B` | seeds per batch | 128 |
| `M_rows` | wide rows per seed (cap) | 128 |
| `C` | cells per wide row (cap; rel-f1 flattened ≈ 20–32) | 48 |
| `d` | model width | 256 |
| `d_sig` | signature width | 64 |

Grid tensors (built in collate):

* `cell_grid`   `[B, M_rows, C, d]`  — encoded cells (zeros at pad)
* `cell_mask`   `[B, M_rows, C]`     — bool, real cell
* `row_mask`    `[B, M_rows]`        — bool, real wide row
* cell tags     `[B, M_rows, C]`     — `col_id / table_id / path_id`, and `recency_bin`
* row tags      `[B, M_rows]`        — base `table_id / fk_role (relation to seed) / hop`, and `recency_bin`

---

## 3. The model, step by step (with shapes)

Every attention block below is **pre-norm MHSA + `MoEFFN`** (reuse `gloss/setjoin/model.py::MoELayer`),
routed on a value-free signature (`route_on ∈ {signature, hidden, dense}` — same arms as today).

**Step A — cell embedding.** Reuse `CellEncoder.encode_type` verbatim (value dtype-encoder + frozen name
token) and scatter into the grid → `cell_grid [B, M_rows, C, d]`. Add cell tags (path/recency; markers for
absent parents) as today's wide path does.

**Step B — cell encoder (cell-level MHSA + MoE).** Reshape to `[B·M_rows, C, d]`; run `L_cell` MoE layers
where the `C` cells of a row attend to each other (columns attend within a row); pad-safe (re-zero pads
after each sublayer, as `AxialCellBlock` already does). Router reads the **cell signature** `[B, M_rows, C,
d_sig]` = `RMSNorm(name + modality + recency + path)` (reuse `WideSignature.type_cells`). → `[B, M_rows, C, d]`.

**Step C — RowPool (slot-attention, cells→row).** For each wide row, pool its `C` cell embeddings to one
row embedding via slot attention (`n_cell_slots` slots attend over cells, softmax over slots, measure/mean
aggregate, combine to 1). → **`row_emb [B, M_rows, d]`**. (Replaces the gated `RowPool`; the slot machinery
adapts `SlotReadout`.)

**Step D — row encoder (row-level MHSA + MoE).** Run `L_row` MoE layers over the `M_rows` row embeddings
(rows attend to each other), pad-masked by `row_mask`. Router reads the **row signature** `[B, M_rows,
d_sig]` = `RMSNorm(table + fk_role + recency + hop)` (reuse `ElemSignature`). → `[B, M_rows, d]`.

**Step E — aggregate (rows→prediction).** `mean` over valid rows (default) → `[B, d]`; **or** `slot` with
`n_slots = out_dim` learned queries → `[B, out_dim, d]` → `[B, d]` (for `out_dim=1`).

**Step F — head.** `LayerNorm → Linear(d→d) → GELU → Linear(d→out_dim)` → **`logits [B, out_dim]`**.
`aux = Σ ortho_loss` over all MoE layers (cell + row encoders), weighted `λ_ortho` as today.

---

## 4. What we build / reuse

New file `gloss/setjoin/row_model.py` (`RowModel`) + new collate `to_row_set_batch` in `collate.py`
(or a sibling module). Reuse, unchanged:

* `CellEncoder.encode_type` (cell value+name), `MoEFFN`, `MoELayer` (both encoders), the pad-safe axial
  re-zeroing idiom, `WideSignature`/`ElemSignature`/`recency_bins`, `slot_relations` (if slot-agg), the head
  MLP, and the whole train/eval scaffolding (`SetJoinLitModule`, `evaluate_split`, `runner`, `grid`) — the
  Lit module just swaps `SetJoin` → `RowModel` and `to_join_batch` → `to_row_set_batch`.

The **sampler is unchanged** (`setjoin_neighbors`, temporal, disjoint). Only the collate's *packing*
changes: instead of (wide cells `[B,W]`) + (pre-pooled set `[B,N]`), emit the cell grid `[B, M_rows, C]`.

---

## 5. Execution — rung by rung (each ends green on its tests before the next)

* **R1 — collate.** `to_row_set_batch` → the `[B, M_rows, C]` grid + masks + cell/row tags. Tests: row-set
  membership (seed row present; one row per direct child; hop-2 excluded), **the seed + m2o parents are
  repeated into every child row** (the OBT denormalization, §0), cell placement on the dual-FK fixture,
  temporal leakage (no cell with `row_time > seed_time`), cap/truncation bookkeeping.
* **R2 — model forward.** `RowModel` (Steps A–F), `aggregate=mean`, `route_on=signature`. Tests: forward
  shapes over all arms, grad flow to router + both signatures, overfit-one-batch (binary + regression),
  permutation invariance over rows and over cells-within-row, empty-children seed finiteness, `dense` arm
  aux==0.
* **R3 — aggregate arm.** `aggregate ∈ {mean, slot}`; `n_cell_slots` cascade knob. Tests: slot-agg grads,
  `out_dim` slots, mean≡slot(1-query) sanity is *not* required (different pools) but both finite.
* **R4 — wire + gate.** Point `SetJoinLitModule`/`runner`/`grid` at `RowModel`; launch the 9-task ×3-seed
  gate at the standing backbone (harrier), TEST-set, → `results/rowmodel/`. Compare vs today's SetJoin
  (`run_setjoin.py --compare`). **Stop & report at the gate.**

**DoD:** `pytest` green (new hermetic tests on the dual-FK fixture + rel-f1-guarded forward); a `--dry-run`
prints the grid shapes + param count (≤30M); the gate array launches.

---

## 6. Non-goals / guardrails

Keep **one mechanism** (the signature-routed `MoEFFN`) — it is the only learned novelty; the hierarchy is
plumbing. Router stays **value-free** (no neighborhood/global stat). Temporal leakage safety is a hard,
tested invariant. Do **not** delete the old `SetJoin` until the gate adjudicates. No new datasets, no
text-encoder training. `mean`-aggregate ≈ old model is a valid honest outcome.
