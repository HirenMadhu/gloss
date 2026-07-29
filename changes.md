# changes.md — Two-level temporal relational encoder

Target: replace the flat cell-token RT substrate with a two-level (cell, row)
substrate that carries time and schema structure into attention.

Scope of this document is the **encoder only**. No LLM, no projector, no TabQA.
Evaluation is RelBench entity tasks on rel-f1, rel-trial, rel-stack via the
existing linear head.

Read `report.md` (the codebase investigation) before starting. Every file:line
reference below is from that report against commit `2403be4` on `main`; verify
they still hold.

> **`report.md` is not in the repository** as of this branch. Every `report X##`
> citation below is therefore currently unverifiable — the empty-bucket
> statistics (E16), the 97–98% masked-FLOP figure (D14), the post-truncation row
> maxima behind `R = 160`, and the router-collapse evidence (F17–F19). Locate it
> or re-derive the numbers before treating any of them as established.

---

## 0. Scope

### In

| Item | Why |
|---|---|
| Row tokens (`[ROW]` per sampled row) | Rows currently have no representation; masks are the only encoding of row structure |
| Continuous fixed-frequency time ladder replacing 20 log-decade buckets | 14 of 20 buckets empty, 3 carry 94% of cells (report E16) |
| Temporal RoPE at cell level | Δ currently reaches only the router (report C9, C10) |
| Temporal RoPE at row level | same; replaces the additive relative-time bias of earlier drafts (§3.5) |
| Name-derived FK role + hop biases at row level | `fk_role_id` built then discarded, hop never computed (report C11) |
| Zero dataset-fitted constants in the model | the encoder is meant to become a foundation model; see the rule below |
| Single full cell attention replacing four masked ones | 97–98% of attention FLOPs on masked/padded pairs (report D14) |
| Cross-attention row encoder (hybrid query) | replaces mean-pool; addresses Griffin App. C.1 degeneracy |
| Row-level MoE, dense cell-level FFN | cell router is a 129-key lookup table (report F17, F19) |
| Head reads the seed row token | replaces mean over 6 seed cells (report H21) |
| Fanout sweep | survived degree hard-capped at 12 for every high-degree relation (report B8) |

### Out (do not implement in this run)

- Sigmoid gating and all six moment channels (m, S, μ, ς, x, e)
- Exact degree / windowed-degree prefix-count features
- Statistic tokens over the unsampled neighborhood
- Task-conditioned queries as an **active** input (keep the signature slot, feed
  a constant; it is provably a no-op in per-task training)
- Box / conjunctive-predicate kernel
- LLM projector, LoRA, TabQA
- `identity` and random-partition router controls
- Absolute calendar-phase channels (hour-of-day / day-of-week / day-of-year
  sin-cos). Universal across databases and the right home for genuine
  periodicity, but a new feature, not part of this run.
- Any normalisation statistic derived from the sampled subgraph or the batch.

Anything in the Out list that appears in a diff is a bug.

### The no-dataset-artifact rule (governs every section below)

This encoder is intended to become a foundation model: pretrain on some
databases, run on a schema never seen during training. That imposes one rule,
and it is the tiebreaker whenever a design choice is otherwise close.

> A tensor may depend on the data only if it is **recomputable from a new
> database without gradients**. A tensor may **not** be a constant fitted on the
> training corpus, nor have a shape indexed by a training-set id.

Legal: the frozen Qwen name embeddings (regenerate from the new schema's
strings); hop, direction, modality (small universal integers); any learned
parameter whose shape is fixed by `d_model` / `d_sig` / `n_freq`.

Illegal: a checkpointed `μ`/`σ` fitted on the training split; `γ ∈ R^{2K+2}`
indexed by role id; an embedding sized to the stypes that happen to occur in one
bundle. All three appeared in earlier drafts of this document; §3 and P0.5 now
remove them. Test §6 "no dataset-specific artifact" is the standing guard.

---

## 1. Prerequisites (P0) — do these first, nothing else works without them

All in `collate.py` / `graph.py`. Phases 2–4 depend on every item here.

### P0.1 Preserve FK role on the edge

`collate.py:206-219` currently appends only the parent's local index and drops
which FK column produced the reference. `graph.py:145-152` builds `fk_role_id`
and `metapath_id`; both are unused by the model.

Record, per (row, edge), the role id. Define role ids over the global schema:

```
role_id : (child_table, fk_column, parent_table) -> int in [1, K]
```

Table-pair granularity is wrong: two FK columns in the same child table can
reference the same parent table and must stay distinct.

### P0.2 Build the row-level adjacency

New `CellBatch` fields. `R = max_rows_per_seed`, config, assert not exceeded.
Observed post-truncation row maxima: rel-f1 69, rel-trial 63, rel-stack 139.
Set `R = 160` and assert.

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `num_rows` | `[B]` | int64 | rows actually present per seed |
| `row_valid` | `[B,R]` | bool | slot holds a real row |
| `row_table` | `[B,R]` | int64 | table id (pad −1) |
| `row_time` | `[B,R]` | float64 | row timestamp (0 if untimed) |
| `row_is_timed` | `[B,R]` | bool | |
| `row_hop` | `[B,R]` | int64 | BFS distance from seed root, root = 0 |
| `row_in_role` | `[B,R]` | int64 | role by which this row was reached (root = 0) |
| `row_is_root` | `[B,R]` | bool | exactly one True per seed |
| `adj_role` | `[B,R,R]` | int64 | see below |
| `cell_row` | `[B,S]` | int64 | alias of existing `node_idxs`, kept for clarity |

`adj_role[b,r,s]` encoding:

```
0                 no edge
1 .. K            s is a CHILD of r via role s->r   (reverse FK)
K+1 .. 2K         s is a PARENT of r via role r->s  (forward FK)
2K+1              r == s (self loop)
```

Admissibility mask for row attention is `adj_role != 0`, AND-ed with
`row_valid` on both axes. Self-loops always present so no row is fully masked.

Derive from `f2p_nbr_idxs` plus P0.1: all cells of a row share the same row, so
collapse `f2p_nbr_idxs [B,S,max_fk]` to per-row, then symmetrise with the role
direction flipped.

### P0.3 Hop

BFS from the root over `adj_role != 0`, or read PyG's
`num_sampled_nodes_per_hop` from the loader with `disjoint=True`. Assert
`row_hop <= len(num_neighbors)` and `row_hop[row_is_root] == 0`.

### P0.4 Name embeddings for tables and roles

`schema.py:21-23` builds the column-name string. Add, using the same frozen
Qwen3-Embedding-4B path:

- table name: `"table {t}"`
- role name: `"table {child} column {fk_col} references table {parent}"`

Store as `table_name_emb [n_tables, d_text]`, `role_name_emb [K, d_text]`.
Name-derived rather than learned per-id, so an unseen schema works on day one.

**Encoder — DECIDED: `qwen` (Qwen3-Embedding-4B, `d_text = 2560`).** Assert at
startup that `name_emb.shape[-1] == 2560` and fail loudly otherwise.

Two facts this branch inherits, both verified against the cache on disk:

- The cache contains **no hash encoder**. `data/schema_cache/` holds `qwen`
  (2560) and `harrier` (`microsoft/harrier-oss-v1-27b`, 5376) only. The report's
  "hash encoder" environment note is stale or refers to instrumentation runs.
- **The existing baselines are `harrier`, not qwen** (`recap.md:44`;
  `run_gridsearch.py` header). Coverage is uneven: rel-f1 has both, rel-trial
  has harrier only, rel-stack has qwen only — no encoder covers all three.

Consequences of choosing qwen, both mandatory:

1. **Build the rel-trial qwen cache** before any 9-task run.
   `scripts/build_schema_cache.py` defaults to qwen and is content-hash keyed,
   so re-running over all three datasets computes only what is missing.
2. **Every number in `recap.md` is a harrier number and is not a valid
   reference for this branch.** Phase 0a's acceptance ("within 1 std of the
   current RT+MoE numbers") therefore needs a **fresh qwen baseline of the
   current architecture on all 9 tasks × 3 seeds** first. Budget it: without it,
   Phase 0a's gate compares against a different encoder and means nothing.
   This baseline doubles as the §6 parity reference — capture it before P0.5.

### P0.5 Pin the modality (stype) id space

A live bug, not a hypothetical. `schema.py`'s `build_column_modality_ids`
docstring states the id space is *"the set of stypes actually present in this
bundle (sorted by name for determinism), not a fixed enum — so `stype_emb` is
sized to the dataset."* That makes both the id *meanings* and `stype_emb`'s
*shape* dataset-dependent: `numerical` can be id 1 on rel-f1 and id 2 on
rel-stack, and a checkpoint trained on one bundle silently mis-indexes on
another.

Pin to the full pytorch-frame stype enum in a fixed, declared order, with
`n_stypes` a constant independent of the bundle. Unknown/absent stypes simply
get an unused row. Cheap now; silently corrupting later.

### P0.6 Tests for P0

- `adj_role` is consistent: `adj_role[b,r,s] in [1,K]` iff
  `adj_role[b,s,r] in [K+1,2K]`.
- Every non-root valid row has at least one edge.
- `row_time[b,r] <= seed_time[b]` for all valid `r`. This is the leakage assert;
  it must never be relaxed.
- Row count from `adj_role` matches `num_rows` matches
  `node_idxs.unique()` per seed.
- Modality ids are bundle-independent: the id of a given stype is identical
  across all three datasets, and `n_stypes` is constant.

---

## 2. New modules

```
gloss/model/time_encoding.py   fixed frequency ladder, tau(), rotate(), feats()
gloss/model/row_level.py       RowPool, RowAttention, Broadcast, RowSignature
gloss/model/two_level.py       TwoLevelBlock, TwoLevelSubstrate
gloss/data/row_graph.py        adj_role / hop / row_* construction (or in collate.py)
```

`rt_substrate.py` stays untouched and reachable so Phase 0 can A/B against it.

---

## 3. Math

Batch index suppressed. Cells `i ∈ [1,S]`, rows `r ∈ [1,R]`, `ν(i)` = row of
cell `i`, `c(i)` = global column id, `k(r)` = table of row `r`. Seed time `t*`.

### 3.1 Time encoding (`time_encoding.py`)

Time enters the architecture at **exactly one point**: a rotation of `q`/`k`, at
the cell level and at the row level. There is no Time2Vec, no learned frequency,
and no fitted statistic anywhere in the time path.

**Canonical unit.** Δ is measured in **seconds**, always, on every database:

$$\Delta_r = \max(0,\; t^* - t_r)\ \text{[seconds]}, \qquad
\tau_r = \log(1 + \Delta_r)$$

Seconds is a physical unit, not a fitted one. Every database that has ever
existed lands in `τ ∈ [0, 22]` — one second is 0.69, one century is 21.9. The
range is known before a single row is read, so there is nothing to calibrate.
**No standardisation, no `μ_τ`, no `σ_τ`, no per-database unit, no calibration
file.** Earlier drafts of this document specified a checkpointed `(μ_τ, σ_τ)`;
that is exactly the illegal artifact of §0 and it is gone.

Two facts make the calibration unnecessary rather than merely inconvenient:

- **`μ_τ` cancels exactly.** RoPE scores depend on `τ_i − τ_j`; an additive
  shift is invisible to the model. In a rotation-only design the mean was dead
  weight.
- **`σ_τ` is replaced by a wide fixed band.** Rather than squashing each
  database to unit variance, the frequency ladder spans the whole `[0, 22]`
  range at multiple resolutions and each database reads whichever channels
  carry signal for its own spread. This is the standard multi-resolution RoPE
  argument, and it is why RoPE generalises across sequence lengths without
  per-corpus statistics.

**The ladder.** `n_freq = 8` frequencies `ω_k`, log-spaced over
`[ω_min, ω_max] = [0.05, 5.0]`, registered as a **non-persistent buffer, not a
parameter** — fixed constants, never learned:

$$\theta^{(k)}_r = \omega_k\,\tau_r$$

Band design: the lowest frequency has period `2π/0.05 ≈ 126` in τ units, so it
is monotonic across the entire 0–22 range with no wraparound; the highest has
period `≈ 1.26`, resolving recency ratios of about `e^{1.26} ≈ 3.5×`. Verify
both ends against the observed Δ histogram before committing the constants.

**Untimed rows.** `θ = 0` (all channels), plus a **learned scalar** `b_untimed`
added to the attention logit whenever either endpoint is untimed. One universal
parameter; no dataset dependence. `θ = 0` alone would collapse untimed rows onto
"Δ = 0, maximally recent", which is wrong, hence the explicit flag.

**Two readouts of the same ladder.** Rotation (§3.2, §3.5), and — where a
feature vector is needed rather than a rotation (§3.3) — the fixed pair
`[sin θ_r ; cos θ_r] ∈ R^{2·n_freq}` behind a *learned linear map*. The learned
linear map is legal under §0; only the frequencies were ever the risk, and they
are now constants.

Delete `_RECENCY_EDGES` (`signature.py:27`) and `recency_bins`
(`signature.py:45-50`), superseded by the continuous ladder. Keep the old path
behind `time.mode: buckets` for one A/B arm, then remove. Note for the write-up:
those buckets were already unit-robust by construction (a fixed absolute decade
grid); the ladder must not lose that property, which is what test §6
"time-unit invariance" enforces.

### 3.2 Cell level

Cell embedding is unchanged (`column_encoder.py:100-101`):

$$x_i = W_v \operatorname{Enc}_{\text{dtype}}(v_i) + W_{\text{name}}\,\text{name}_{c(i)}$$

**Temporal RoPE.** Apply to the first `2m` dims of each head (`m = d_h/4`) using
the §3.1 ladder (`n_freq = 8`, fixed, `[0.05, 5.0]`):

$$\theta_i^{(k)} = \omega_k \, \tau_{\nu(i)}, \qquad
\tilde q_i = \mathcal{R}(\theta_i)\, W_q h_i, \qquad
\tilde k_j = \mathcal{R}(\theta_j)\, W_k h_j$$

so the score depends on

$$\tau_{\nu(i)} - \tau_{\nu(j)} = \log\!\big((1+\Delta_i)/(1+\Delta_j)\big),$$

a log time **ratio**. This is the property the whole design rests on: rescaling
a database's entire time axis by a constant `c` leaves every score unchanged for
`Δ ≫ 1s`, because `log(cΔ_i) − log(cΔ_j) = log Δ_i − log Δ_j`. Scale-equivariance
by construction, with no fitted constant doing the work. The invariance is exact
in that regime and approximate as `Δ → 0`, where the `+1` floor bites; §6 tests
it at `Δ ≥ 1 day` and states the regime rather than overclaiming.

Untimed rows get `θ = 0` and the `b_untimed` logit term of §3.1.

**Attention.** One full attention with a padding mask only:

$$h \leftarrow h + \operatorname{Attn}\big(\tilde q,\, \tilde k,\, W_v h;\ \text{pad mask}\big)$$

`col`, `feat`, `nbr` are deleted. `feat` and `nbr` are absorbed by the row
level; `col` is dropped (see §5, Phase 0b acceptance).

A padding-only mask is passable as `is_causal=False` with a bool mask, but
prefer building the batch so real cells are left-packed and using
`attn_mask=None` with variable-length packing where the kernel allows it. The
point of this change is to get off the math backend; verify with a profiler
that flash / mem-efficient SDPA is actually selected.

FFN: dense SwiGLU, unchanged.

### 3.3 Row signature (`RowSignature`)

Value-free, name-derived, transfers to unseen schemas:

$$s_r = \operatorname{RMSNorm}\big(
W_{\text{tab}}\,\text{name}_{k(r)} +
W_\rho\,\text{name}_{\rho_{\text{in}}(r)} +
W_\eta\,\operatorname{emb}(\eta_r) +
W_\tau\,[\sin\theta_r\,;\,\cos\theta_r]\big)
\in \mathbb{R}^{d_{\text{sig}}}$$

`d_sig = 128`. Computed once per forward, reused by every block. Used by the
row encoder query and the row MoE router.

The recency term is the §3.1 ladder read out as features, `W_τ ∈ R^{d_sig × 2·n_freq}`
learned, `ω` fixed. Every term is name-derived, a small universal integer, or a
fixed basis — nothing is indexed by a training-set id, so `s_r` is defined on an
unseen schema. The same substitution applies to the **cell** signature
(`signature.py`), whose recency term likewise moves from 20 bucket embeddings to
this readout; the router still routes on `(column, modality, recency)` as
CLAUDE.md requires, just through a continuous fixed basis instead of bins of
which 14 of 20 were empty.

### 3.4 Low to high: cross-attention row encoder (`RowPool`)

Keys are column-name embeddings, values are cell states (Griffin's split).
Query is hybrid, `M` slots:

$$q_r^{(m)} = W_Q^{(m)}\big[\,u_r \,;\, s_r\,\big], \qquad m = 1..M$$

$$a^{(m)}_{r,i} = \operatorname{softmax}_{\,i:\,\nu(i)=r}\!\left(
\frac{q_r^{(m)\top}\big(W_K\,\text{name}_{c(i)}\big)}{\sqrt{d_h}}\right)$$

$$u_r \leftarrow u_r + W_O\Big[\textstyle\sum_i a^{(m)}_{r,i}\,W_V h_i\Big]_{m=1}^{M}$$

`M = 4`. Query arm is a config flag: `mean` (no cross-attention, plain mean over
the row's cells) / `signature` (`q = W_Q s_r`) / `hidden` (`q = W_Q u_r`) /
`hybrid` (above). Deliberately mirrors the existing `route_on` vocabulary.

Row token init at block 0:

$$u_r^{(0)} = W_u\big[\,\bar h_r \,;\, s_r\,\big]$$

with `h̄_r` the mean of the row's cell embeddings.

### 3.5 Row level attention (`RowAttention`)

Over `R ≤ 160` rows, so a dense `[B,H,R,R]` float bias is affordable
(64·8·160·160·4B ≈ 52 MB, acceptable; drop to fp16 or reduce R if not).

Time enters here by **rotation, not by an additive bias** — the same mechanism
and the same ladder as the cell level:

$$\tilde q^{(h)}_r = \mathcal{R}(\theta_r)\,W_q^{(h)} u_r, \qquad
\tilde k^{(h)}_s = \mathcal{R}(\theta_s)\,W_k^{(h)} u_s$$

$$s^{(h)}_{rs} = \frac{\tilde q^{(h)\top}_r \tilde k^{(h)}_s}{\sqrt{d_h}}
\;+\; \gamma^{(h)}_{rs}
\;+\; b_{\text{untimed}}\,\mathbb{1}[\,r\ \text{or}\ s\ \text{untimed}\,]$$

$$u_r \leftarrow u_r + \operatorname{Attn}\big(s^{(h)};\ \text{mask} = (\text{adj\_role} \ne 0)\big)$$

Softmax. No gate in this run.

**Role bias, name-derived.** The earlier `γ^{(h)} ∈ R^{2K+2}` indexed by
`adj_role` is removed: `K` is fixed by the training schemas, so on an unseen
database the role ids are new and `γ` is *undefined* — unrecoverable, not merely
miscalibrated. It also contradicted P0.4, which builds `role_name_emb` precisely
so roles are name-derived. Instead:

$$\gamma^{(h)}_{rs} = \big\langle v^{(h)},\, W_\rho\,\text{name}_{\rho(r,s)}\big\rangle
\;+\; c^{(h)}_{\operatorname{dir}(r,s)},
\qquad \operatorname{dir} \in \{\text{child},\ \text{parent},\ \text{self}\}$$

with `v^(h) ∈ R^{d_sig}` and `c^(h) ∈ R^3`. `adj_role == 0` stays masked; the
role id is now only an *index into the frozen name table*, never a parameter
index. Same in-distribution capacity, three learned directions instead of
`2K+2`, and no parameter shaped by `K`.

**On the additive time bias that was here.** `⟨w^(h), φ̃_rs⟩` could express a
*content-independent* temporal prior ("recent rows matter more, whatever is in
them"); RoPE modulates the `q·k` interaction and structurally cannot. That is a
real, if small, expressivity loss and it is deliberate — pure rotation keeps time
to one mechanism at both levels. Note the artifact was never the additive
*form*, it was the learned `ω` and fitted `μ, σ`: an additive bias over the
**fixed** ladder, `b^{(h)}_{rs} = ⟨w^{(h)}, [\sin;\cos](\theta_r - \theta_s)⟩`,
has zero dataset constants and is legal under §0. It is retained as the Phase 1
`t3b` arm (`row.time_bias: fixed_basis`) so the term is deleted with evidence
rather than by assumption.

### 3.6 High to low: broadcast (`Broadcast`)

$$h_i \leftarrow h_i + W_b\, u_{\nu(i)}$$

FiLM variant behind a flag, off by default.

### 3.7 Row-level MoE

$$z_r = s_r, \qquad
\text{logits}_e = \frac{1}{T}\,
\frac{\langle W_g^{(e)},\, z_r\rangle}{\|W_g^{(e)}\|\,\|z_r\|}, \qquad
g = \operatorname{softmax}_{\text{top-}k}(\text{logits})$$

$$u_r \leftarrow u_r + \sum_e g_e\, E_e\big(\operatorname{RMSNorm}(u_r)\big)$$

`M_exp = 4`, `k = 2`, `T` learned scalar init 1.0. Cosine routing because the
orthogonality penalty constrains row directions but leaves logit scale free
(report F18: aux ≈ 0.57 at init, all row norms 0.55–0.62).

Auxiliary losses, both on:

$$\mathcal{L}_{\text{aux}} = \lambda_{\text{ortho}}\sum_{\ell}\big\|\hat W_g \hat W_g^\top - I\big\|_F^2
\;+\; \lambda_{\text{bal}}\, M_{\text{exp}}\sum_e f_e\, p_e$$

`f_e` = fraction of rows with top-1 `e`, `p_e` = mean gate probability for `e`.
`λ_ortho = 0.5` (existing), `λ_bal = 0.01`.

Cell-level FFN stays dense. Cell-level MoE is a Phase 4 ablation arm only.

### 3.8 Head

$$\text{logit} = W_2\,\operatorname{GELU}\big(W_1\,\operatorname{LayerNorm}(u_{\text{root}})\big)$$

`u_root` selected by `row_is_root`. Replaces the mean over `is_seed_cell`
(`heads.py:27-32`). Keep the old head behind `head.mode: seed_cells` for the
Phase 0 parity check.

### 3.9 Block order

Per block `ℓ = 1..L`:

```
1. cell attention        (temporal RoPE, pad mask)
2. cell FFN              (dense SwiGLU)
3. low->high  RowPool
4. row attention         (role bias, relative-time bias)
5. row FFN               (MoE)
6. high->low  Broadcast
```

Six sublayers per block, up from five. Sweep `L ∈ {4, 6, 8}`; do not assume 8.

---

## 4. Config

Add a `model.two_level` block. Every item below is a switch so each phase is
independently ablatable.

```yaml
model:
  arch: two_level            # two_level | rt   (rt = current, for A/B)
  n_blocks: 6
  d_model: 256
  d_sig: 128
  max_rows: 160

  cell:
    attention: full          # full | four_mask
    rope_time: true
    rope_dims: 16            # 2m per head
    ffn: dense

  row:
    pool_query: hybrid       # mean | signature | hidden | hybrid
    pool_slots: 4
    role_bias: name_derived  # name_derived | none   (id_lookup REMOVED, see 3.5)
    time_bias: rope          # rope | none | fixed_basis
    ffn: moe                 # dense | moe
    router: cosine           # cosine | linear
    num_experts: 4
    top_k: 2
    lambda_ortho: 0.5
    lambda_balance: 0.01

  broadcast: additive        # additive | film | none

  head:
    mode: row_token          # row_token | seed_cells

time:
  mode: rope                 # rope | buckets   (buckets = current, A/B only)
  unit: seconds              # canonical, not configurable per dataset
  n_freq: 8
  omega: [0.05, 5.0]         # FIXED, log-spaced, buffer not parameter
  causal_paths: false        # Phase 1 arm

data:
  collate:
    seq_len: 512
  sampler:
    num_neighbors: [12, 12]
```

---

## 5. Phases

Each phase has an acceptance criterion. Do not start the next phase until the
current one passes. Every phase reports mean ± std over 3 seeds on all 9 tasks.

### Phase 0a — row tokens, cell level unchanged

`arch: two_level`, `cell.attention: four_mask`, `cell.rope_time: false`,
`time.mode: buckets`, `row.role_bias: none`, `row.time_bias: none`,
`pool_query: mean`, `row.ffn: dense`, `head.mode: row_token`.

`time.mode: buckets` here on purpose: Phase 0a isolates the row-token addition,
so it must keep the *current* recency encoding. The ladder enters at Phase 1.

Isolates the row-token addition alone.

**Accept:** within 1 std of the current RT+MoE numbers on all 9 tasks.

### Phase 0b — collapse cell attention

Flip `cell.attention: full`.

**Accept:** (a) within 1 std of Phase 0a; (b) profiler confirms flash /
mem-efficient SDPA is selected, not the math backend; (c) measured wall-clock
per step drops. Report the FLOP reduction as a number.

If (a) fails, the `col` mask was load-bearing. Add a cell-level same-column
attention back as a third operator and record that the decomposition is three
operators, not two. Do not silently paper over it.

### Phase 1 — time (headline)

Arms, cumulative:

| Arm | Config |
|---|---|
| `t0` none | Phase 0b |
| `t1` ladder features in the signature only | `time.mode: rope`, `rope_time: false`, `row.time_bias: none` |
| `t2` + cell RoPE | `cell.rope_time: true` |
| `t3` + row-level RoPE | `row.time_bias: rope` |
| `t3b` fixed-basis additive row bias instead | `row.time_bias: fixed_basis` |
| `t4` + path-causal masking | `causal_paths: true` |

`t3b` is the one arm that tests the content-independent temporal prior discussed
in §3.5. It is legal under §0 (fixed frequencies, no fitted constants). Run it
only if `t3 > t2`; if time helps at all, it is worth knowing whether rotation
alone captures it.

Before building `t4`: determine whether the PyG sampler filters child times
against the **parent** or only against the **seed**. If only against the seed,
paths in the subgraph are not internally time-ordered and `t4` is a sampler
change, not a mask change. Document the finding either way.

**Accept:** `t3 > t0` on a majority of the 9 tasks, outside seed variance.
This is the paper's main result. If it fails, say so and stop; do not go
hunting for a configuration where it works.

Read TGAT (Xu et al., ICLR 2020) before writing this section up. It puts a
functional time encoding inside attention on temporal graphs and is the closest
prior art.

### Phase 2 — schema structure

| Arm | Config |
|---|---|
| `s0` | Phase 1 winner |
| `s1` + role bias | `row.role_bias: name_derived` |
| `s2` + hop in signature | `row_hop` term in `s_r` |

Requires P0 in full.

**Accept:** report whichever of `s1`, `s2` helps; a null here is a publishable
negative given that the information was being discarded.

### Phase 3 — row encoder

`pool_query ∈ {mean, signature, hidden, hybrid}` × `pool_slots ∈ {1, 4}`.

**Accept:** report the table. The interesting comparison is `hybrid` vs
`hidden` at block 0, which is the Griffin App. C.1 degeneracy question.

### Phase 4 — routing

| Arm | Config |
|---|---|
| `r0` dense | `row.ffn: dense`, matched width (see below) |
| `r1` row MoE | `row.ffn: moe` |
| `r2` cell MoE instead | `cell.ffn: moe`, `row.ffn: dense` |

**Matched control.** The MoE combine is dense (every expert runs, gates only
weight them), so active FFN width is `num_experts × d_ff`. The honest `r0` is a
single SwiGLU at `4 × d_ff`, not `d_ff`. Report both `dense@d_ff` and
`dense@4·d_ff`.

Log from step 1: per-block expert usage histogram, `H(expert | table)`,
`H(expert | role)`, `W_g` row norms, both aux terms. The current setup cannot
distinguish collapse from intent; this is what fixes that.

**Accept:** `r1 > r0@4·d_ff` outside seed variance, or the MoE gets one line in
the ablation table and no subsection.

### Phase 5 — fanout

With Phase 0b's freed FLOPs. Sweep `num_neighbors ∈ {[12,12], [24,12], [32,16]}`
at `seq_len ∈ {512, 1024}`.

Measure pre-truncation cells per seed **before** committing to a config, and
report the binding fraction. rel-f1's true p90 1-hop degree is 19.7, so 24
covers most of the distribution; rel-stack's true degrees reach 32k and nothing
in this run helps it.

**Accept:** report the accuracy/fanout curve. This is a strong figure regardless
of which direction it goes.

---

## 6. Tests

New tests, all must pass before Phase 0a is declared done.

**Leakage (never relax these)**
- `row_time[b,r] <= seed_time[b]` for every valid row.
- `τ_r >= 0` for every timed row (`Δ` is clamped at 0, so `log1p(Δ) >= 0`).
- The existing `test_routing_invariance.py` and `test_signature.py` still pass
  with the continuous ladder. They must: `ω` is a constant and `τ` depends only
  on the cell's own timestamp and the seed time, so the signature remains
  invariant to which neighbours were sampled.

**Row graph**
- `adj_role` symmetry with direction flip.
- Exactly one `row_is_root` per seed.
- `row_hop[root] == 0`, `max(row_hop) <= len(num_neighbors)`.
- `num_rows` agrees across `adj_role`, `row_valid`, and `node_idxs.unique()`.

**Time encoding**
- *Relative-only*: for fixed `h`, the attention logit between cells `i, j`
  depends on `τ_i` and `τ_j` only through `τ_i − τ_j`. Test by adding a constant
  to all `τ` and asserting logits are unchanged to 1e-5. This is what makes the
  removed `μ_τ` provably redundant.
- *Time-unit invariance* (the foundation-model test). Multiply every timestamp
  in the fixture by `c ∈ {60, 86400}` and assert RoPE logits are unchanged to
  1e-5 **for `Δ ≥ 1 day`**. State the regime: the invariance is exact for
  `Δ ≫ 1s` and approximate near the `+1` floor. Do not assert it at `Δ → 0`; the
  approximation there is correct behaviour, not a defect.
- Untimed rows: `θ = 0`, the `b_untimed` path is exercised, and it is *not*
  equivalent to `Δ = 0` (assert the logits differ). rel-trial has 27% untimed
  cells; rel-stack has 0%, so test on rel-trial.
- `τ` and the ladder are finite for `Δ = 0` and for `Δ = max` observed, and
  `τ ∈ [0, 22]` on all three datasets — the universal range of §3.1. A value
  outside it means Δ is not in seconds.

**No dataset-specific artifact** (the standing guard for the FM claim)
- Walk `state_dict()`: assert no entry's shape depends on `K` (role count), on
  `C` (column count), or on the per-bundle stype count, and that no buffer holds
  a statistic fitted on the data. `ω` is present but constant — assert it is
  byte-identical across two models built on different datasets.
- Assert `stype_emb.num_embeddings` and every stype id are identical across
  rel-f1, rel-trial and rel-stack (P0.5).
- A model built on rel-f1 loads a rel-trial bundle's schema tables without any
  shape mismatch. This is the cheapest possible proxy for the deferred LODO
  transfer run and it costs seconds.

**Shapes**
- Every new `CellBatch` field matches the table in §1.2 in shape and dtype, on
  all three datasets.
- `R = 160` is not exceeded on any seed of any dataset (assert, don't clamp).

**Parity**
- A `arch: rt` run reproduces the pre-change numbers bit-for-bit given the same
  seed. This is the regression guard for the whole refactor.

---

## 7. Instrumentation (always on, logged per epoch)

- Attention: which SDPA backend was selected, per block.
- Mask/pad density and effective FLOPs, cell and row level.
- Pre-truncation cells per seed, post-truncation rows, binding fraction.
- `τ` distribution per dataset (mean, std, min, max) and the fraction of cells
  with `τ` outside the band the ladder actually resolves. On the training DBs
  this should be ~0; on a transfer DB a nonzero value is the miscalibration
  alarm firing *before* a run is wasted. This line replaces the deleted
  calibration record as the thing that catches scale mismatch.
- Ladder channel utilisation: per-frequency `|∂ logit / ∂ θ^{(k)}|` or, cheaply,
  the variance of `sin θ^{(k)}` across cells. A channel with near-zero variance
  is outside the data's range and the band constants should move.
- Row-level bias magnitudes: `|γ|` per role and per direction. If the role bias
  has near-zero magnitude, Phase 2 is a null for a mechanical reason and you want
  to know that immediately. Same for `|b^{(h)}|` on the `t3b` arm.
- Expert usage histogram, `H(expert | table)`, `H(expert | role)`, `W_g` row
  norms, both aux terms.
- Seed variance across the 3 runs per config, reported alongside every number.

---

## 8. Evaluation protocol

Three databases, nine tasks (confirm the exact task list against the RelBench
release you are pinned to before starting):

- rel-f1: driver-dnf, driver-top3, driver-position
- rel-trial: study-outcome, study-adverse, site-success
- rel-stack: user-engagement, user-badge, post-votes

Three seeds per config. TEST metrics are the reported result; select on VAL.
Report mean ± std always. Deltas in this literature are frequently smaller than
seed variance (RelGNN's published gains over HeteroGNN on entity classification
run 0–1%), so a table without error bars is not a result.

27 runs per arm. Budget accordingly and prune arms aggressively at each phase
gate rather than running the full cross product.

---

## 9. Resolve before starting

1. **Name encoder.** Confirm the reported baseline numbers were produced with
   Qwen3-Embedding-4B and not the hash encoder. The report's environment note is
   ambiguous. Add the startup assertion from P0.4 regardless.
2. **Sampler temporal semantics.** Does `NeighborSampler(time_attr="time",
   temporal_strategy="last")` filter a child's time against its parent or
   against the seed? Determines whether Phase 1 `t4` exists.
3. **Task list.** Pin the RelBench version and confirm the nine tasks.
4. **`R = 160`.** Verify against a full epoch of each dataset, not a sample. The
   report's rel-stack figures are over 2,560 seeds, not the full 1.36M.
5. **Row-level bias memory.** `[B,H,R,R]` at `B=64, H=8, R=160` is ~52 MB fp32
   per block. Confirm it fits alongside activations on the A40, or move to fp16
   / reduce `R`.
6. **Timestamp units — RESOLVED: already UNIX seconds.** RelBench's
   `to_unix_time` (`relbench/modeling/utils.py:11-27`) floor-divides ns by 1e9,
   and `collate.py:155,230` carries the result through as float64. §3.1's
   `τ = log1p(Δ)` therefore needs **no conversion**, and rel-f1's ~70-year span
   gives `τ ≈ 21.5`, inside the `[0, 22]` band. Keep the §6 range assert anyway:
   `to_unix_time`'s integer-dtype branch passes integers through *unconverted*,
   so a future dataset whose `time_col` is a raw integer in another unit would
   silently arrive in the wrong scale.
7. **Parity vs P0.5 — ordering constraint.** Pinning the stype enum changes
   `stype_emb.num_embeddings`, which changes init RNG draw order, which breaks
   the §6 bit-for-bit parity guard. Capture the `arch: rt` parity baseline
   **before** P0.5 lands, or the strongest regression guard in the plan is
   unavailable for the rest of the refactor.
8. **Ladder constants.** `[0.05, 5.0]` and `n_freq = 8` are derived from the
   0–22 range, not measured. Check them against the observed Δ histogram per
   dataset before Phase 1 and adjust if the resolved band misses the mass.
