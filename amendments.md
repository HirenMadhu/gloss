# amendments.md — what changed on `multi-level`, and why

Companion to [changes.md](changes.md). `changes.md` is the **plan**; this file is the **record of
where the plan was wrong**, what was measured to establish that, and what is actually built.

Every number here was measured in this repo, not cited. `report.md` — which `changes.md` was
written against — is **absent**, so all of its figures were unverifiable; re-deriving them is how
most of the findings below surfaced.

Scripts that produce these numbers:
`scripts/measure_substrate.py`, `scripts/probe_sampler_causality.py`,
`scripts/probe_fk_role_collision.py`, `scripts/probe_relevent_time.py`.

---

## 0. Status at a glance

**Built:** the P0 prerequisites layer (minus P0.4/P0.5), a standalone time-encoding module, a
bit-for-bit parity guard, and leaderboard comparison against RT + GelGT. **124 tests green.**

**Not built:** the two-level architecture itself. No phase has run. No model consumes any of the
new row-level fields, and `TimeLadder` is imported nowhere.

| item | state |
|---|---|
| P0.1 role vocabulary (triple-keyed) | ✅ done, tested |
| P0.2 `adj_role` row adjacency | ✅ done, tested |
| P0.3 hop (BFS) | ✅ done, tested |
| P0.4 table/role name embeddings | ❌ **not started** |
| P0.5 pinned stype enum | ❌ **not started** (unblocked now — see §7) |
| `gloss/model/time_encoding.py` | ✅ exists, 21 tests — **wired into nothing** |
| `gloss/model/row_level.py` | ❌ missing |
| `gloss/model/two_level.py` | ❌ missing |
| §3.2–3.9 (cell RoPE, single attention, RowPool, RowAttention, Broadcast, row MoE, row-token head) | ❌ missing |
| §9.7 parity baseline | ✅ done, guard proven to fail on init-order change |
| Leaderboard comparison (RT + GelGT) | ✅ done |
| Phases 0a–5 | ❌ none run |
| 27-run qwen baseline | ❌ not launched (qwen caches now built) |

---

## 1. `changes.md` claims that measurement **falsified**

### 1.1 `R = 160` — safe, but the source figures were wrong in both directions

| dataset | task | mean | p90 | p99 | **max** | `report.md` claimed |
|---|---|---|---|---|---|---|
| rel-f1 | driver-dnf | 34.0 | 50 | 63 | **65** | 69 |
| rel-trial | site-success | 11.9 | 25 | 25 | **25** | 63 |
| rel-event | user-attendance | 32.1 | 48 | 65 | **100** | — |

`R = 160` holds, but at **1.6× margin**, not the "generous" one an earlier draft claimed, and the
assert is **fanout-coupled** — Phase 5's `[32,16]` raises the ceiling roughly proportionally and
would fire it on rel-event.

**Incidental finding worth more than the number:** rel-trial's rows/seed has
p90 = p99 = max = **25 = 1 + 12 + 12**. Every above-median seed hits the `[12,12]` cap *exactly*, so
the row count reports the **sampler's fanout, not the data**. That is the degree-capping concern
behind Phase 5, confirmed by measurement rather than citation.

### 1.2 The masked-FLOP fraction has **no single value**

| dataset | seq_len | col | feat | nbr | full | mean | wasted |
|---|---|---|---|---|---|---|---|
| rel-f1 | 512 | 1.03% | 1.05% | 0.52% | 20.17% | 5.69% | **94.31%** |
| rel-f1 | 1024 | 0.31% | 0.31% | 0.18% | 5.09% | 1.47% | 98.53% |
| rel-trial | 512 | 0.66% | 1.44% | 0.21% | 14.10% | 4.10% | **95.90%** |
| rel-event | 512 | 0.74% | 9.74% | 0.34% | 52.47% | 15.82% | **84.18%** |

The report's "97–98%" reproduces **only at `seq_len=1024`**, the collate *default* — while
`changes.md` §4 pins `512`. At 512 the honest range is **84–96%**, driven by padding: rel-event
averages 306 real cells of 512 where rel-f1 averages 209.

Three corrections to how Phase 0b must be argued:

1. **Density is not the speedup.** Four attentions → one is exactly **4×** fewer score matrices
   regardless of density. The density figure supports a *different* claim — that the masks buy
   little coverage. Report them separately; do not let one imply the other.
2. **Varlen packing matters more than the mask collapse.** Even collapsed, the single remaining
   attention is only ~20% useful pairs at 512 on rel-f1.
3. See 1.3.

### 1.3 `seq_len=512` is **binding**, not loose

| dataset | mean cells/seed | p90 | max | truncated at 512? |
|---|---|---|---|---|
| rel-f1 | 209 | 290 | 353 | no — 0% of seeds |
| rel-trial | 150 | 329 | 329 | no — 0% of seeds |
| rel-event | **306** | **512** | **512** | **yes — ≥10% of seeds** |

rel-event's **p90 sits exactly on the cap**. The "truncation binds on 0% of seeds" claim was
generalised from the only two DBs then measured. Seed-row cells are emitted first and survive, so
the loss is *neighbour* context — but **rel-event is silently running at a different effective
fanout than the other two**, and Phase 5's premise (freed FLOPs buy fanout headroom before
`seq_len=1024` is needed) fails for it.

### 1.4 P0.1 is a role-**vocabulary** fix, not plumbing

`changes.md` framed P0.1 as "`collate.py` drops the role id, plumb it through." The ids it would
plumb were already collapsed: `graph.py` keyed `fk_role_id` on the **FK column name alone**, so
same-named FK columns in different child tables merged.

| dataset | FK edges | roles before | roles after | worst collision |
|---|---|---|---|---|
| rel-f1 | 13 | **4** | 13 | `raceId` merged 5 relations |
| rel-trial | 15 | **6** | 15 | `nct_id` merged **10** |
| rel-event | 7 | **4** | 7 | 3 names merged 2 each |

**rel-trial is the consequential one.** All ten child tables reach `studies` via a column named
`nct_id`, so all ten relations shared one id. A role bias on that vocabulary cannot distinguish
`outcomes → studies` from `sponsors_studies → studies` — **Phase 2's `s1` arm would have returned a
null for a purely mechanical reason**, and been read as "schema structure doesn't help."

Scoping, both verified: `fk_role_id` was consumed **nowhere** in `gloss/` outside `graph.py`, so the
bug was latent and no existing number depends on it; and the existing
`test_dual_fk_relations_get_distinct_ids` passed because it checks two *differently-named* columns.
`changes.md` was internally consistent — P0.4's role-name string is already the triple. Only the
code was behind.

### 1.5 `t4` is a **sampler** change, not a mask change

`changes.md` §5 said to determine, before building `t4`, whether the sampler filters a child's time
against its **parent** or only against the **seed**.

| dataset | hop-adjacent timed pairs | child EARLIER | EQUAL | child **LATER** | verdict |
|---|---|---|---|---|---|
| rel-f1 | 5,939 | 0% | **100%** | 0% | uninformative |
| rel-trial | 2,950 | 0% | **100%** | 0% | uninformative |
| rel-event | 9,038 | 12.8% | 32.0% | **55.3%** (worst +197.6 d) | **seed-only** |

**The sign breakdown is what makes this trustworthy.** rel-f1 and rel-trial both report zero
violations — which reads as a clean "parent-filtered" confirmation — but *every* pair shares a
timestamp, so they prove nothing about sampler policy. Only rel-event can distinguish the two, and
it says **seed-only**: paths inside a sampled subgraph are **not** internally time-ordered.

Consequence: `t4` cannot be built as a mask. Masking cannot impose an ordering the sample never
had. It is a sampler change — materially more work than its one-line config entry suggests.

### 1.6 The RoPE ladder band `[0.05, 5.0]` is **wrong**

| dataset | τ mean | τ **std** | dead channels (`var(sin ωτ) < 0.01`) |
|---|---|---|---|
| rel-f1 | 16.25 | **1.61** | ω = 0.050, 0.097 |
| rel-trial | 17.57 | **1.59** | ω = 0.050, 0.097 |
| rel-event | 14.03 | **1.90** | ω = 0.050 |

The band was derived from τ's **range** `[0, 22]`. What the frequencies must resolve is τ's
**spread** — and every seed sits at τ ≈ 14–18 with σ ≈ 1.6–1.9. The two lowest channels vary by
< 0.1 rad across the entire corpus: **2 of 8 channels are dead weight** on two of three DBs.
Healthy channels begin around ω ≈ 0.35–0.7. Re-derive `ω_min` from σ before Phase 1.

This does not breach §0 — the frequencies remain fixed constants chosen once from corpus-wide
statistics, not per-database fitted parameters. But **say so explicitly in the write-up**, because
"chosen by looking at the data" invites precisely that objection. §7's per-frequency utilisation
logging is now known to be load-bearing, not nice-to-have.

---

## 2. Bugs found and fixed

### 2.1 Missing timestamps encoded as the year 1677 (`collate.py`)

RelBench's `to_unix_time` maps pandas `NaT` to `pd.Timestamp.min.value // 1e9` = **-9223372037**,
which reads as **1677-09-21**. The sentinel is *finite*, so nothing downstream flagged it — and
`collate.py` set `is_timed = True` for every row of a timed **table**. A row with no timestamp
therefore encoded as *"≈336 years before the seed"* rather than *"time unknown"*.

Confirmed by exact count match on rel-event:

| table | NaT | negative timestamps |
|---|---|---|
| `event_attendees` | 64,918 | **64,918** |
| `users` | 58 | **58** |
| `events` | 3 | 4 ← one genuinely corrupt row |

`events` also reaches **2222-02-02** — future garbage, excluded from neighbour sampling by the
`time_attr` filter and clamped to Δ = 0 if it reaches a cell.

**Fix:** a per-**row** plausibility gate (floor 1800-01-01 = `-5364662400`, below rel-f1's genuine
1950 data and above the sentinel). Such rows become `is_timed = False`, `row_time = 0` —
**reclassified, never dropped**, so what gets sampled is unchanged and comparability with the
benchmark is preserved.

**This is how τ = 23.08 was caught**, which means the §6 `τ ∈ [0, 22]` assert earned its place.
`changes.md` framed that assert as a *units* check ("a value outside it means Δ is not in
seconds"). It is a broader **validity alarm** — widen the rationale, don't narrow the assert.

> **Corpus consequence:** ~65k rel-event rows previously carried a bogus τ ≈ 23. Every
> pre-existing rel-event number was computed under the buggy handling. Treat the 27-run qwen
> baseline as the **first valid rel-event reference**; do not mix it with older rel-event results.

### 2.2 The leakage assert fires on a third of rel-event

`changes.md` P0.6/§6 said flatly `row_time[b,r] <= seed_time[b]` for all valid `r`. On real data
that is wrong twice, and both gates are load-bearing:

- **`row_is_timed`** — untimed rows carry sentinel `0`, and real UNIX seconds go *negative*
  (rel-f1 starts in 1950), so an ungated compare flags every untimed row.
- **`~row_is_root`** — the root **is** the query entity, and RelBench includes its row regardless
  of its own timestamp. Measured on rel-event/`user-attendance`: **35.0%** of task rows have the
  `users` row's own time *after* the seed time (median +9.7 d, worst +152 d).

Unscoped, the assert fires on a third of rel-event and reads as a bug in our code. Neighbour
leakage — the thing that would actually be wrong — remains fully covered, and
`test_planted_row_leak_is_detectable` still requires exactly one planted leak to be caught.

### 2.3 `build_schema_cache.sh` never sourced `env.sh`

The only SLURM script here that didn't (`prep.sh`, `run_ablation.sh`, `run_gridsearch.sh` all do).
Two consequences, both silent:

- `GLOSS_SCHEMA_CACHE` fell back to the repo-relative `data/schema_cache/`, which is **not** what
  training jobs read (`$HOME/scratch60/gloss/schema_cache`). The job would "succeed" and produce
  nothing usable.
- `HF_HOME` fell back to `$HOME/.cache`, putting an 8 GB model download on the home filesystem
  instead of scratch.

Also `--gpus=a40:1` pinned the job to the single a40 node on the `gpu` partition — the actual
reason it sat `PENDING (Resources)`. Relaxed to `--gpus=1`; it started immediately on an h100 and
finished in 29:53.

**Downstream correction:** the qwen files in `data/schema_cache/` (rel-f1, rel-stack) are dev
leftovers in the fallback directory. The real cache held **harrier only, for all three DBs — zero
qwen**. An earlier coverage table in this branch was read from the fallback path and was wrong.

### 2.4 `adj_role` cannot be derived from `f2p_nbr_idxs`

`changes.md` P0.2 says to derive row adjacency by collapsing `f2p_nbr_idxs`. PyG records a
traversed edge **only under the edge type it was traversed in**, so the forward `f2p_*` store is not
the transpose of the sampled `rev_f2p_*` store. On one rel-f1 batch at `[12,12]`: 616 forward-store
edges vs 1,476 reverse-store. Forward-only leaves **482 of 1,162 rows — every hop-2 row —
unreachable from its seed** (run and confirmed).

`adj_role` is therefore built from **every edge type, both directions**. `f2p_nbr_idxs` is left
untouched (frozen RT input, parity), which incidentally means RT's `feat`/`nbr` masks see only
**61%** of cells with any parent link — a pre-existing latent gap, flagged, not fixed.

Also: relbench's `CustomNodeLoader` does **not** forward PyG's `num_sampled_nodes`, so hop must come
from BFS over the sampled subgraph. There is no loader attribute to use.

### 2.5 DB3 is rel-event, not rel-stack

`eval/ablation.py:LEADERBOARD_TASKS` has no rel-stack entry and never did; every run this repo has
executed used rel-f1 / rel-trial / rel-event. `changes.md` said rel-stack throughout, and two of its
conclusions were derived from that error — the P0.4 encoder argument and §6's untimed-cell test.
Task list is now pinned to `LEADERBOARD_TASKS`: rel-event is `user-repeat` / `user-ignore` /
`user-attendance`.

---

## 3. Resolved without change

- **§9.5 row-bias memory — the worry was aimed at the wrong level.** `[B,H,R,R]` at
  `B=64, H=8, R=160` fp32 is 52.4 MB/block. The **cell** attention at `S=512` is 537 MB for a
  *single* score matrix — `(512/160)² = 10.2×` larger, and there are currently four per block. Row
  attention is ~2% of a block's attention memory. No fp16, no reduced `R`.
  Implementation contract: compute γ as a per-`(head, role)` scalar table `[H, K]` once per forward
  and **gather** it into the score tensor. Never materialise γ as a `[B,H,R,R]` parameter — `K` must
  not enter any weight shape (§0).
- **§9.3 task list** — pinned, see 2.5.
- **§9.6 timestamp units** — already UNIX seconds; no conversion needed.
- **Leaderboard** — `results/leaderboard_baselines.json` already held RT + GelGT for all 9 tasks;
  a live re-scrape showed **zero drift**. The gap was plumbing: only the RT-only file was read, so
  GelGT never reached a comparison table. Now fixed, with NMAE = MAE/train-std pinned in code.

---

## 4. Open design questions (deliberately not decided)

1. **Seed-row Δ clamps to τ = 0 — and it is COUPLED to the ladder band, which was not obvious.**
   Because 35% of rel-event seeds have the root dated after the seed time and
   `Δ = max(0, t* − t_r)`, those seed cells get **τ = 0 — indistinguishable from "happened right
   now."** "This is the query row" and "this is maximally recent" are different statements.
   Structurally the same argument §3.1 makes for `b_untimed`; same remedy available (a learned root
   flag).

   **The coupling.** That τ = 0 mass is the *only* reason the re-derived band `[0.3, 5.0]` still
   trips a wraparound check. On rel-f1 the τ **bulk** runs 11.4 → 19.66 (span 8.2, `ω·span = 2.47
   rad < π`, no wrap), but the full observed **range** is 0 → 19.66 (span 19.7, `5.90 rad`, wraps) —
   entirely because of the clamped seed rows. So while §9.10 is undecided, the lowest ladder channel
   aliases **τ ≈ 0 against τ ≈ 21** — the query row against an ancient row. Fixing §9.10 removes the
   outlier mass and the band becomes clean with no further tuning. Decide it in Phase 1, and note it
   is now a correctness issue for the time encoding, not only a modelling nicety.

   Note also: τ's distribution is **truncated on the right** (τ cannot exceed the database's age), so
   the bulk is not a symmetric ±3σ interval. An earlier draft of the band derivation assumed symmetry
   and wrongly concluded `[0.3, 5.0]` was over-constrained; it is not.
2. **The MoE moves a level.** §3.7 puts the MoE on the **row** FFN and makes the **cell** FFN dense.
   CLAUDE.md's "route on the value-free relational signature" survives; its "the MoE enters at
   exactly one point — RT's SwiGLU FFN inside each `RelationalBlock`" does not. Phase 4's `r2` arm
   keeps today's cell-MoE as the comparison, so the question is *which level*, not *whether*.
3. **`feats` under untimed.** θ = 0 makes `[sin;cos] = [0;1]`, identical to Δ = 0, and `b_untimed` is
   an attention-logit term that never reaches the §3.3 row signature — so the signature would
   confound "untimed" with "maximally recent". Currently the whole pair is zeroed (linearly
   separable behind `W_τ`). The alternative is an explicit indicator channel, which widens `W_τ` to
   `2·n_freq + 1`.
4. **§6's 1e-5 tolerance is not scale-free.** The `+1`-floor residual in the relative angle is
   `ω_max/(1+Δ_min)` = 5.8e-5 rad at Δ = 1 day; the resulting *logit* deviation scales with
   ‖q‖‖k‖ and is **4.7e-5** with unit-variance q,k in d=32 — the spec's own assertion fails, in
   float64 and float32 alike, so it is the encoding's approximation, not numerics. Tests row-normalise
   q,k and add a scale-free angle-bound companion. §6 should state the q/k scale it assumes.
5. **§3.1 contradicts itself** on `_RECENCY_EDGES`: "delete it" and, one sentence later, "keep the
   old path behind `time.mode: buckets` for one A/B arm." Untouched; needs a ruling.
6. **`rope_dims: 16` and `n_freq: 8` coincide only because `d_h = 32`.** If head count changes,
   `2m ≠ 2·n_freq` and §3.2 doesn't say which frequencies to drop. Currently drops the lowest.

---

## 5. Infrastructure notes

- **qwen caches built** (job `29029249`, 29:53 on h100): rel-f1 `(45, 2560)`, rel-trial
  `(103, 2560)`, rel-event `(122, 2560)`, in the real scratch60 cache.
- **`run_ablation.sh` needs two fixes before the 27 runs**: it hardcodes `--encoder harrier`, and it
  targets `--partition=gpu_h200`, which **does not exist** on this cluster (`sinfo`: `gpu` with
  a40/h100, `priority_gpu`, `scavenge`). `prep.sh` has the same stale partition.
- **`results/` is gitignored**, so baseline data is not in the repo and must be regenerable —
  hence `scripts/fetch_leaderboard.py`.
- **`R = 160` is a live assert on the current training path** but nothing yet consumes the row
  fields, so a `max_rows` passthrough should land with the first consumer.

---

## 6. The encoder decision, recorded

P0.4 says `qwen`. That was chosen when DB3 was believed to be rel-stack and when the coverage table
had been read from the wrong directory. On the corrected facts — **harrier covers all three DBs
today, qwen covered none** — the decision was re-put and **qwen was reaffirmed**, so the costs are
accepted, not overlooked:

- an 8 GB model download plus three cache builds (**done**);
- a fresh **27-run** qwen baseline of the *current* architecture, because every number in
  `recap.md` is a harrier number and Phase 0a's "within 1 std of the current RT+MoE numbers" has no
  valid qwen reference otherwise.

The declined alternative, recorded so the tradeoff stays legible: on harrier,
`results/v2_add_signature/*_signature.json` is **already** exactly those 27 records (9 leaderboard
tasks × 3 seeds, base `signature` router, current architecture), so the Phase 0a gate would have
cost zero GPU-hours. Nothing in §0's foundation-model rule distinguishes the two encoders — both are
frozen name tables, both recomputable on an unseen schema.

---

## 7. What unblocked what

- **P0.5 was blocked by §9.7** and is now free. Pinning the stype enum changes
  `stype_emb.num_embeddings`, which changes init RNG draw order, which breaks the bit-for-bit parity
  guard — so the baseline had to be captured **first**. It has been, and verified: P0.1 and P0.2 did
  *not* disturb parity (role ids and the new row fields reach no weight shape). When P0.5 lands,
  expect 7 parity tests to fail; that is **correct**, and the baseline should be re-captured with
  `--force` **in the same commit**.
- **The parity guard is proven, not assumed.** Simulated P0.5 (`n_stypes` 2 → 12) → 7 failures, with
  64/95 initial parameters differing bit-for-bit. A pure init-order change with no shape change
  (one extra `torch.randn(1)`) → 5 failures, while the shape/count test *passed* — which is exactly
  the case shape checks alone cannot catch.
- **P0.4 is the next real blocker for the foundation-model claim.** Without `table_name_emb` /
  `role_name_emb` there is no way to represent an unseen table or role at all. The NaT fix (2.1)
  removes a silent-failure hazard on new schemas but is **not** a generalisation mechanism.
