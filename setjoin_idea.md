# SetJoin — join semantics without join materialization

*A standalone research direction (branch `setjoin`). The MoRE stack on `main` is untouched; this
document is normative for the branch, alongside [setjoin_implementation.md](setjoin_implementation.md).*

## Thesis

A relational prediction problem "wants" to be a single-table problem: classical practice denormalizes
the database into one big joined table and trains a tabular model on it. Two things break that story:

1. **One-to-many joins duplicate the seed row.** Joining a parent entity against its N child rows
   yields N copies of the parent — N predictions for one label. Aggregating the children first (mean,
   count, DFS-style feature engineering) fixes the duplication but is lossy, fixed a priori, and
   destroys the ability to predict on the child side.
2. **Temporal tasks make the big table a moving target.** Every training example is
   `(entity, seed_time)`; the correct join is *as-of* `seed_time`. There is no single static joined
   table — the same entity has a different joined row at every task timestamp.

**SetJoin keeps the join semantics and moves the aggregation into the model:** a relational
neighborhood is represented as **one wide joined row plus one set of fact rows**, and a **set
transformer** (learned, task-conditioned, permutation-invariant) replaces both the row duplication and
the hand-picked aggregate.

- The **wide seed row** is the *many-to-one closure* of the seed: the seed's cells, its parents'
  cells, its grandparents' cells — every FK hop on the many-to-one side contributes **at most one
  row**, so flattening is exact and duplication-free. This is precisely the fragment of the universal
  join that is safe to materialize per example.
- The **union set** holds the rows a SQL flat join would have duplicated the seed against: all
  one-to-many child rows, from *all* child relations, in **one** set. Each element is one child row —
  itself widened by flattening in *that row's* parents (again exact) — tagged with which relation it
  arrived through, which table it is, and how long before `seed_time` it happened.
- Seed-side tasks (the common case — all 9 leaderboard tasks sit on the "one" side) pool the set with
  a seed-conditioned attention readout. Leaf-side tasks degenerate gracefully: the set is empty and
  the model predicts from the wide row alone (a learned null element keeps attention well-defined).

The as-of join is never materialized: RelBench's leakage-safe temporal neighbor sampler
(`time_attr="time"`, `temporal_strategy="last"`) already yields exactly the as-of-correct rows per
seed, so SetJoin is a **collate + model**, not a data pipeline.

## The mechanism

Per seed: cell-encode every relevant row with the frozen-LM-name + stype value encoder (shared with
RT/MoRE), then

```
seed_repr = TabularTransformer( wide seed row cells + join-path tags | z_wide )   # CLS readout
E_i       = RowPool(child_i cells) + Σ RowPool(child_i's parents) + tags          # tags: relation, table, Δt-recency
context   = PMA_{seed_repr}( SetAttention({E_i} ∪ {null} | z_elem) )              # seed-conditioned pooled readout
ŷ         = MLP([seed_repr ; context ; log1p(child counts)])
```

No relational attention masks, no graph message passing at depth — one flat row, one set, one pooling.

**Every layer's FFN is MoRE's Mixture-of-Relational-Experts** (`MoEFFN`, reused verbatim): the top-k
router reads a **value-free signature** while the experts transform the hidden state — "route on
semantics, transform the content", carried onto the single-table substrate. Wide tokens ARE cells, so
they route on the true MoRE cell signature, `z_wide = RMSNorm(W_s·name_c + ψ(modality_c) + φ(Δt) +
π(join path))`; set elements are rows, so they route on the row-level analog, `z_elem =
RMSNorm(table + FK role + φ(Δt) + hop)`. Balance is the router-orthogonality loss (`aux`, weighted by
`λ_ortho`), not a uniform load-balancing loss. Routing arms: `signature` (the method) | `hidden`
(router reads the normed hidden state) | `dense` (plain FFNs, aux=0) — `signature vs dense` is the
in-substrate headline, and the dense arm is exactly the v2 gate.

## Positioning

- **vs RT / MoRE (this repo's main line):** RT contextualizes every cell against every cell through
  four relational masks over a 512-token soup — O(S²) attention, deep. SetJoin is the shallow, cheap
  cousin: structure is baked into *where* a cell lands (wide row vs set element) rather than learned
  through masked attention. The scientific question: **how much of RT's deep relational attention does
  a one-hop set-pooled wide table recover, at a fraction of the compute?**
- **vs DFS / feature-engineering denormalization:** same "one big table" instinct, but the aggregate
  is learned and task-conditioned instead of fixed (mean/max/count), and child-side granularity is
  preserved up to the set representation.
- **vs GNNs:** the union set + pooling is architecturally a single message-passing step with attention
  aggregation. SetJoin differs in the m2o closure being *flattened, not message-passed* (exact join,
  no averaging over parents) and in being seed-centric (no node updates, no depth).
- **vs DeepSets / Set Transformer:** those are the pooling machinery; SetJoin is their application to
  the relational-join duplication problem with a principled wide-row/set split of the schema.

## What would falsify it

- **SetJoin ≈ the mean-pool baseline** (`eval/baselines.py::seed_features` + LightGBM): if learned set
  attention doesn't beat a count + column-means aggregate, the set transformer adds nothing and the
  honest result is negative.
- **SetJoin ≪ RT (from scratch)** across the board: the 2-hop o2m context RT sees (children-of-children,
  siblings) is what the MVP deliberately drops; a large uniform gap says relational depth, not join
  framing, is what matters on these tasks.

Reference points (per task): `RT (from scratch)` and `GelGT` from
`results/leaderboard_baselines.json`; MoRE grid-search best (recap §3).

## Non-goals / scope guards (MVP)

- No SQL / DuckDB / materialized join tables — the temporal sampler is the join engine.
- Union set = **1-hop children only**; children-of-children / sibling context deferred (`elem_hop`
  field reserved =1). This is the first extension axis, not an oversight.
- No sparsity/efficiency claims beyond the obvious (no O(S²) cell attention).
- Only the 9 RT-reported leaderboard entity tasks (rel-f1, rel-trial, rel-event), 3 seeds, TEST-set
  reported; binary AUROC, regression NMAE (= MAE / train-std).
- ≤ ~30M params. Known approximations, documented in the implementation spec: fanout + set-size caps
  truncate heavy entities; a missing parent (NULL FK / sampler miss / temporal exclusion) is one
  marker token in the wide row and silently absent inside a set element.
