# UPDATE: Relational Slot Attention — a differentiable GROUP BY for the o2m readout

**Branch:** `setjoin`
**Blast radius:** small — this changes **only the final pooling step** of the union-set encoder
(the PMA readout). The set self-attention / MoE layers, the wide-row encoder, the cell encoder,
the recency bins, the routing signatures, and the head are all **unchanged**.

**Instruction to the implementer (Claude Code):** read the files below *first* and match their existing
conventions (names, config plumbing, tensor layouts, masking idioms). This spec fixes the **math and the
interface**; do not treat the pseudocode variable names as literal. Implement **rung by rung** (§5) — get
Rung 1 running and passing its acceptance check before touching Rung 2. Do **not** build all three rungs
at once, and do **not** remove or alter the existing PMA / `route_on` arms / head / recency code.

Files to read before editing:
- `gloss/setjoin/model.py` — the union-set encoder and the current PMA readout
- `gloss/setjoin/collate.py` — `JoinBatch` construction (this is where the group index is added)
- `gloss/setjoin/recency.py` — recency bins (only needed for the optional recency-grouping key)
- `gloss/model/moe.py` — `MoEFFN` (read-only context; not modified)

---

## 1. Why (the one-paragraph thesis)

The union-set readout currently pools child rows with a **softmax PMA** — a convex combination whose
weights sum to 1, which is **invariant to the number of children**. Relational prediction is
overwhelmingly *aggregation over groups* (COUNT/SUM/AVG of children, often per relation, often
conditional). Softmax pooling structurally cannot represent that; the codebase already half-admits it by
routing raw per-relation counts straight to the head. This update replaces the readout with a
**differentiable GROUP BY**: slots seeded by the schema's grouping keys, elements assigned to slots by
key-match softened by content, and each group pooled by a **cardinality-aware measure** (keep the count,
don't normalize it away). Every deviation from generic Slot Attention is derived from a property of
relational data, which is what makes this a mechanism rather than a bolt-on.

---

## 2. The crucial design insight (read this before implementing)

Generic Slot Attention has **two** normalizations. Only one is wrong for us.

1. **Assignment** (per element, distribute over the `K` slots): Slot Attention uses `softmax` **over
   slots**. This is *fine* — it is a `K`-way softmax (K = number of slots, fixed), **not** an N-way
   softmax over elements, so it introduces **no** cardinality-invariance. It gives a clean soft
   partition (each element is fully assigned across groups). Keep it.
2. **Within-slot aggregation** (each slot pools its assigned elements): Slot Attention uses a **weighted
   mean** — it divides by the column-sum of assignment weights. **This is the count-blind step.** A group
   of 3 and a group of 300 produce the same slot vector.

**The fix is exactly one thing:** do **not** normalize the within-slot aggregation. Instead expose the
unnormalized column-sum as the group **count**, and read SUM / MEAN off the same weights. The mean is
still available (it's `SUM / (COUNT + ε)`), so this is *strictly more* information than Slot Attention.
Because the count comes from summing over elements (unnormalized), it correctly tracks cardinality; the
assignment softmax is over slots and stays harmless.

The relational inductive bias (schema-seeded slots + key-bias) is layered on the **assignment**, and is
orthogonal to the count fix on the **aggregation**.

---

## 3. Contribution surface — three readout arms (make it decomposable)

Add a `readout` axis **parallel to the existing `route_on` axis**, defaulting to current behavior:

| `readout` | assignment | within-group pool | grouping | role |
|---|---|---|---|---|
| `pma` *(default, unchanged)* | softmax over elements | weighted **mean** | none | baseline: count-blind |
| `measure` | `n_pma` sigmoid gates | **measure** (count/sum/mean) | none | isolates *counting* |
| `slot` | softmax-over-slots + key-bias | **measure**, per group | schema (table, fk-role) | isolates *grouping* |

This triple is the point: `pma → measure` credits **cardinality-aware pooling**; `measure → slot`
credits **schema grouping**. The paper can attribute each independently. Implement all three arms; `pma`
must remain reachable and default so every existing result reproduces bit-for-bit.

`slot` has a sub-mode `slot_mode ∈ {hard, soft, iterative}` (the Rung ladder, §5).

---

## 4. The mechanism (precise)

### 4.1 Collate side (`collate.py`)

The union set already carries per element: child-table id, `fk_role_id`, recency bin, hop. Add a
**group index** and the **slot identity table**.

- `K_max` = number of distinct `(child_table, fk_role)` child relations in the schema (a fixed,
  schema-derived constant; small). Compute once from the same schema metadata that already assigns
  `fk_role_id`s. Each `(table, fk_role)` pair → a fixed slot index in `[0, K_max)`.
- Add to `JoinBatch`:
  - `set_group_idx : [B, N]` int — each set element → its slot index in `[0, K_max)`; pad/absent
    elements → `-1` (or a dedicated pad-sink index, whatever matches existing masking).
  - `slot_meta : [K_max, ...]` — slot index → its `(table_id, fk_role_id)` identity, for embedding init.
    This is per-schema (batch-independent); build once and reuse.
  - *(optional)* `set_slot_mask : [B, K_max]` bool — occupied slots per seed. **Not required for
    correctness** (empty slots read as true-zero groups under measure pooling — see §4.4), but useful for
    the collapse diagnostic (§6).

Grouping key is `(table, fk_role)` by default (`slot_group_key=table_fkrole`). A finer key
`table_fkrole_recency` (group = relation × recency-bin, so `K_max × n_recency_bins` slots) is an
extension; the element's recency bin already exists, so `set_group_idx` just indexes the product space.
Default to `table_fkrole` for Rungs 1–2.

**Do not** change anything else in the set assembly. No positional encoding on the set (still a tested
contract — slot assignment is content+key based and order-independent, so permutation invariance holds).

### 4.2 Schema-seeded slots (kills "anonymous, fixed-K slots")

Do **not** sample slots from a Gaussian. Instantiate **one slot per relational group**, initialized from
the group's identity:

```
slot0[g] = W_slot( concat[ T_emb(slot_meta[g].table), F_emb(slot_meta[g].fk_role) ] ) + p_slot[g]
```

`T_emb`, `F_emb` are the same-family learned table / fk-role embeddings already used for element tags;
`p_slot` is a small learned per-slot offset so identical-key slots can differentiate. `K` is now
schema-determined and correct.

**Seed-conditioning** (preserve the existing contract that *what to extract depends on the seed*): add a
projection of `seed_repr` to the slot init:

```
slot[b,g] = slot0[g] + W_seed( seed_repr[b] )        # broadcast over g
```

### 4.3 Assignment: content + schema key-bias (kills "pure content competition")

```
Q = W_q(slot)                      # [B, K, d]
Kk = W_k(H)                        # [B, N, d]   H = contextualized child elements from the set encoder
content = Q @ Kk.transpose         # [B, K, N] / sqrt(d)
keymatch = onehot(set_group_idx, K).transpose(-1,-2)   # [B, K, N]; 1 at (slot(i), i)
logits = content + gamma * keymatch                    # gamma: learned scalar (per head optional)
logits = logits.masked_fill(pad_elements, -inf)        # mask pad/absent elements
```

- `gamma → ∞`: hard GROUP BY (each element → its schema group; content ignored) → **Rung 1**.
- `gamma = 0`: generic content-based slot assignment (no schema prior).
- learned `gamma`: interpolates — mostly schema-grouped, able to pull an anomalous element toward a
  content-appropriate slot. This is the "hand it the schema but let it deviate" knob.

### 4.4 Measure pooling within each slot (kills the count-blind mean)

**Assignment softmax is over the SLOT dimension (`K`), per element — not over elements.** Then keep the
count:

```
if slot_mode == "hard":
    attn = keymatch * valid_mask            # [B, K, N] hard partition, content ignored
else:
    attn = softmax(logits, dim=SLOT)        # <-- OVER SLOTS (dim K), per element
    attn = attn * valid_mask

V   = W_v(H)                                 # [B, N, d]
m   = attn.sum(dim=ELEMENTS)                 # [B, K]     soft COUNT per group   (DO NOT divide it away)
S   = attn @ V                               # [B, K, d]  SUM per group
mu  = S / (m.unsqueeze(-1) + eps)            # [B, K, d]  MEAN per group (stable)
# optional MAX/EXISTS channel: (attn.unsqueeze(-1) * V).max(dim=ELEMENTS)

update = W_o( concat[ mu, log1p(m).unsqueeze(-1), S / log1p(N) ] )   # [B, K, d]
```

Degree-scaling matters: content rides the **normalized** `mu`; the raw count/sum channels are
**log-compressed** (`log1p(m)`, `S / log1p(N)`) so they don't blow up with N (the DeepSets/PNA warning).

**Empty groups are handled by construction:** an unoccupied slot has `m=0, S=0, mu=0` → a true-zero group
summary ("this relation has no children for this seed"). No special-casing; no null-element hack needed.

### 4.5 Iterative refinement (Rung 3 only)

```
for t in range(T):        # T=slot_iters, e.g. 3
    <compute attn, update as above using current slot as query>
    slot = GRU(slot, update)     # per-slot GRU cell, shared across slots
```

The `keymatch` term is fixed across iterations, so refinement moves the **content** pull within the
schema scaffold rather than drifting off it.

### 4.6 Readout → context (and expose slot vectors)

```
slot_vectors = slot                          # [B, K_max, d]   <-- KEEP THIS ACCESSIBLE (see §8)
context = W_ctx( slot_vectors.reshape(B, K_max * d) )   # [B, d_context], parallel to current PMA concat->project
```

`context` must have the **same shape/interface** the head already expects, so the head is untouched.
(If `K_max * d` is large, a seed-conditioned attention pool over the `K_max` slots is an acceptable
alternative to concat-then-project; default to concat.)

**Head:** leave `[seed_repr ; context ; log-counts]` unchanged. Note for later: with per-group counts now
inside `context`, the head's raw log-counts become partially redundant — a **bonus ablation** is whether
they can be dropped ("the encoder now recovers what we previously injected at the head"). Do not drop them
by default.

---

## 5. Staged rollout — implement in this order, one at a time

Each rung has an interpretable yes/no. Do not proceed until the current rung's acceptance check passes.

**Rung 1 — `readout=slot slot_mode=hard` (the floor).**
Hard schema grouping + measure pooling, no learned assignment, no iteration. This is deterministic
per-relation cardinality-aware pooling. *Purpose:* does cardinality-aware **grouping** help on RelBench
**at all**? If this moves nothing, stop — slot attention will not save it.
**Acceptance:** runs end-to-end on ≥1 task without NaN; `context` shape matches; empty groups → zero
vectors; **sanity:** per-group `m` under hard mode equals the raw per-relation count for that
`(table, fk_role)` (they must match — good built-in check).

**Rung 2 — `readout=slot slot_mode=soft` (the actual slot-attention contribution).**
Learned `gamma`, softmax-over-slots assignment, schema-seeded, one shot. *Purpose:* does soft,
content-aware regrouping beat hard schema grouping?
**Acceptance:** trains; at large `slot_gamma_init` behaves ≈ Rung 1 (continuity check); assignment
entropy logged (§6).

**Rung 3 — `readout=slot slot_mode=iterative`.**
Add the GRU, `slot_iters=T`. *Purpose:* does message-passing among group summaries help, or is one shot
enough?
**Acceptance:** trains; `T` configurable; compare against Rung 2 on the count/magnitude tasks.

Also implement **`readout=measure`** (the middle arm, §3) — `n_pma` sigmoid-gated queries with the same
count/sum/mean channels but **no grouping**. It's the same measure math without the group index, so it's
cheap to add once Rung 1's channels exist, and it's what isolates "counting" from "grouping."

---

## 6. Diagnostics & honesty checks (slot attention specifically invites these)

Log these as training metrics from the start — they decide whether the "GROUP BY" claim is *supported* or
merely asserted.

1. **`gamma → ∞` floor (Rung 1) vs soft (Rung 2).** If hard grouping already captures most of the gain,
   the learned soft assignment isn't earning itself and the real contribution is just "per-relation
   measure pooling" (simpler, still a result, but a *smaller* claim). Run Rung 1 early to know which world
   you're in.
2. **Assignment entropy per element:** `H(attn[:, :, i])` averaged over elements. At learned `gamma`,
   elements should mostly land on their schema group. If entropy is high / content dominates, the GROUP BY
   claim is unsupported. Pre-register a threshold.
3. **Per-slot utilization:** mean assignment mass per slot. Detect dead slots (never used) or a dominant
   slot swallowing everything (collapse). Schema-seeding should prevent this; verify, don't assume.

---

## 7. Experimental attribution (do not confound the mechanism with capacity)

- **Independent gates.** Compare `pma` vs `measure` vs `slot` (and the Rung ladder) as an isolated axis,
  holding everything else at the standing config. Do not co-vary readout with other changes.
- **FFN-capacity-matched baseline (recurring project caveat, applies here too).** The `slot`/`measure`
  arms add parameters (slot embeddings, assignment/value projections, GRU). Any win must hold against a
  **parameter-matched** `pma` (e.g. more PMA queries / matched FFN width), not only against the current
  config — otherwise it's a capacity result in a relational costume. A reviewer will check this.
- **Pre-registered falsifiable prediction:** `slot`/`measure` should help **most on grouped-aggregate /
  count / magnitude tasks** and be **~neutral on single-salient-child identity tasks**. Your RelBench
  suite splits along this line (count/position/attendance tasks vs. identity-of-one-child tasks). If the
  gain is **uniform**, that is the tell you found **capacity, not counting** — inspect the matched
  baseline.

---

## 8. Note for the two-fold experiment plan (RelBench + TabQA)

Keep `slot_vectors : [B, K_max, d]` accessible from the readout (not just the pooled `context`). In the
RelBench fold they pool into `context` for the head. In the **TabQA / relational-tokenizer** fold, **the
`K_max` group-summary slots ARE the handful of aggregate tokens fed to the LLM** — cardinality-aware,
schema-typed, and few — instead of dumping all rows into the context window. One mechanism serves both
folds: `context` for the from-scratch predictor, `slot_vectors` as the tokenizer output. Do not bury the
slot vectors inside the pooling step.

---

## 9. Suggested config flags (match the project's existing config style/names)

| flag | values | default | notes |
|---|---|---|---|
| `readout` | `pma` \| `measure` \| `slot` | `pma` | parallel to `route_on`; `pma` preserves current behavior |
| `readout_channels` | subset of `{mean,count,sum,max}` | `{mean,count,sum}` | measure/slot arms |
| `slot_mode` | `hard` \| `soft` \| `iterative` | `hard` | the Rung ladder; only for `readout=slot` |
| `slot_group_key` | `table_fkrole` \| `table_fkrole_recency` | `table_fkrole` | recency-grouping is an extension |
| `slot_iters` | int | `3` | only for `iterative` |
| `slot_gamma_init` | float | `2.0` | initial key-bias strength |
| `slot_gamma_learnable` | bool | `true` | learn `gamma` (per-head optional) |
| `slot_seed_cond` | bool | `true` | seed-condition slot init |

Keep `n_pma` in use for the `pma` and `measure` arms.

---

## 10. Do-not-break checklist

- [ ] `readout=pma` reproduces current results exactly (default path untouched).
- [ ] Set-encoder MoE layers, wide-row encoder, cell encoder, routing signatures, recency bins: unchanged.
- [ ] MoE router stays **value-free** — this readout is *after* the set encoder and touches no routing.
- [ ] Head interface unchanged (`context` same shape); raw log-counts kept by default.
- [ ] No positional encoding on the set path (permutation invariance preserved).
- [ ] Empty sets / empty groups produce finite, true-zero summaries (no null-element special-casing needed).
- [ ] Rungs implemented and validated **one at a time** in the §5 order.
