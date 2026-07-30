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

**Built:** all of `changes.md` §1–§4 — the P0 prerequisites, the time ladder, the row level, the
two-level substrate, the row-token head, the config, and the `MoRE(arch=two_level)` wiring, with a
bit-for-bit parity guard and leaderboard comparison against RT + GelGT. **208 tests green**, and the
two-level forward passes on real rel-f1 data.

**Not built:** no phase has been *run*. The 27-run qwen baseline that Phase 0a's gate compares
against is in flight.

| item | state |
|---|---|
| P0.1 role vocabulary (triple-keyed) | ✅ done, tested |
| P0.2 `adj_role` row adjacency | ✅ done, tested |
| P0.3 hop (BFS) | ✅ done, tested |
| P0.4 table/role name embeddings | ✅ done, qwen-cached for all 3 DBs |
| P0.5 pinned stype enum | ✅ done (`N_STYPES = 10`, bundle-independent) |
| `gloss/model/time_encoding.py` | ✅ wired into both levels |
| `gloss/model/row_level.py` | ✅ done, 32 tests |
| `gloss/model/two_level.py` | ✅ done, 35 tests |
| §3.2–3.9 (cell RoPE, single attention, RowPool, RowAttention, Broadcast, row MoE, row-token head) | ✅ done |
| §4 config, `MoRE(arch=two_level)`, `run_train --arch` | ✅ done, real-data dry run passes |
| §9.7 parity baseline | ✅ done, fired on P0.5 and re-captured deliberately |
| Leaderboard comparison (RT + GelGT) | ✅ done |
| Phases 0a–5 | ❌ none run |
| 27-run qwen baseline | 🟡 **RUNNING** — array `29029474`, 27 tasks, %8, h100 |

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


---

## 8. Later corrections (after the first pass)

### 8.1 The clamped-Δ flag — promoted from nicety to correctness

§4.1 recorded seed-row clamping as an open design question. Measuring the re-derived ladder band
promoted it: the τ = 0 mass created by `Δ = max(0, t* − t_r)` is the **sole cause** of the wraparound
check firing. rel-f1's τ *bulk* spans 11.4 → 19.66 (8.2, `ω·span = 2.47 rad < π`, clean), but its
observed *range* is 0 → 19.66 (19.7, `5.90 rad`, wraps). So the lowest channel aliased **τ ≈ 0 against
τ ≈ 21** — the query row against an ancient row.

Fixed on both paths, because neither alone is sufficient:

- **`b_clamped`**, a learned scalar on the attention logit. The rotation *structurally cannot* express
  it: a clamped row has `θ = 0` exactly like a genuine `Δ = 0`, so its relative angle to every other
  row is identical in both cases. Same argument §3.1 makes for `b_untimed`.
- **An indicator channel** in `feats()` (width → `feat_dim = 2·n_freq + 1`), because `b_clamped` is an
  attention-logit term that never reaches the §3.3 row signature.

`was_clamped()` reads the **raw** times; after the clamp the information is gone. Three states must now
stay mutually distinct — timed/`Δ>0`, untimed (all-zero), clamped (sinusoids identical to `Δ=0`, flag
set) — and a test asserts all three pairwise differ, since the natural implementation collapses two.

### 8.2 Row experts are shared + routed; the two levels deliberately differ

User decision. The row MoE gains an always-on **ungated** SwiGLU, on by default, so the routed experts
only model what is *specific* to a row's signature. The **cell** MoE keeps `use_shared=False` — its
`+S` arm measured as only mildly positive, on regression alone (`recap.md`). That asymmetry is
intentional; do not "harmonise" it. `use_shared=False` stays available at the row level as a Phase 4
arm.

Consequence for Phase 4: `r1` now differs from `r0` in **two** ways — the row level exists *and* it is
shared+routed. Isolating the shared expert needs its own arm.

### 8.3 The §6 artifact guard got teeth

The `state_dict` check now also asserts both time flags are **scalars** (`ndim == 0`). That catches
someone later "improving" `b_clamped` into a per-table or per-role vector — a dataset-shaped parameter
that would silently break cross-schema loading, exactly what §0 exists to prevent.

### 8.4 Parity scoping, verified rather than assumed

I predicted the guard would fire on `b_clamped` and it did **not**. Correct: the fingerprinted model is
`arch='rt'` with `time_mode='buckets'`, so it never constructs a `TimeLadder` and `b_clamped` does not
exist there. The guard fires on changes to the **rt baseline** (P0.5 resizing `stype_emb`) and stays
quiet on changes confined to the two-level path. Both behaviours are right.

### 8.5 `run_ablation.sh` and `prep.sh` could never have scheduled

Both requested `--partition=gpu_h200 --gpus=h200:1`, and `run_ablation.sh`'s own comment asserted "this
cluster has no h100". Both false: `sinfo` shows **no `gpu_h200` partition at all**, while `gpu` carries
a40:4 **and** h100:4. Any array submitted with those scripts would have sat pending forever. Now
`--partition=gpu --gpus=h100:1`.

`run_ablation.sh` also pinned `--encoder harrier` *before* `"$@"`, so `--encoder qwen` only won by
argparse last-wins — a reader would reasonably conclude it ran harrier. Removed; the encoder is now
explicit at the call site.

### 8.6 Two test bugs worth remembering

Both were tests that would have "verified" working code forever without exercising it:

- The shared stub batch gave every cell the **same** timestamp. Then `θᵢ − θⱼ = 0`, a uniform rotation
  preserves inner products, and RoPE is *provably* inert — so any test of it passed vacuously. (That
  inertness **is** the §6 relative-only property; it just makes a constant-time fixture useless.)
- `test_row_token_head` perturbed the root row **uniformly** (`+= 10.0`), which the head's leading
  `LayerNorm` removes by mean subtraction. It now perturbs the *direction*.

### 8.7 Corrections to earlier claims in this file

- The band is **not** over-constrained at `n_freq = 8`. That conclusion assumed a symmetric ±3σ bulk;
  τ is **truncated on the right** (it cannot exceed the database's age), so the real bulk span is 8.2
  and both constraints hold comfortably at `ω_min = 0.3`.
- An earlier encoder-coverage table was read from the repo-relative `data/schema_cache/` fallback
  rather than the scratch60 path SLURM reads. The real cache held **harrier only, zero qwen**.

---

## 9. Failures of the 2026-07-29 arrays, diagnosed 2026-07-30

61 of the 243 submitted array tasks failed. Two unrelated causes, plus a third problem that the
failures happened to expose.

### 9.1 `R = 160` did NOT hold — it killed every rel-event run (57 failures)

§1.1 above concluded "`R = 160` holds, but at 1.6× margin", from a measured max of **100** on
rel-event/user-attendance. That measurement was an **undersample**. In the real runs rel-event
reached **161–162** rows on a seed and tripped the assert:

    AssertionError: max_rows_per_seed exceeded: 162 rows on some seed but R=160

All 57 were rel-event; all three of its tasks; both grids and the headline array. It always fired
at the **first validation pass**, i.e. after 5–25 min of training, so every rel-event run was
wasted. rel-f1 and rel-trial were unaffected — which is precisely the trap: `MAX_ROWS = 160` was a
**rel-f1** measurement (max 65 rows/seed at `[12,12]`) frozen into `collate.py` as if it were a
cross-dataset constant. R is fanout- **and schema**-coupled: it grows with `num_neighbors` *and*
with the number of edge types, and rel-event has far more of the latter.

Two things made it undiscoverable short of crashing a job:
1. `configs/*.yaml` advertise `data.collate.max_rows`, but `train/loop.py` never passed it to
   `to_cell_batch` — so the assert's own advice ("Raise `data.collate.max_rows`") was a no-op.
2. §1.1's margin came from a partial sweep, and the shortfall is only ~1%. No amount of extra
   sampling makes a fixed constant safe for the *next* database.

**Fix (`e9d98a1`): `max_rows=None` — fit R to each batch.** R becomes the largest per-seed row count
actually present. This is semantics-preserving, because padding rows are already fully masked
everywhere they could matter — row attention masks them, `RowMoE.balance_loss` and the usage
diagnostics restrict to `row_valid`, and `RowTokenHead` pools the root only — and it *saves* memory:
`adj_role` and row attention are dense `O(R²)`, so padding rel-f1's 65 rows out to 160 cost ~6×. An
explicit int is still honoured as a hard cap that asserts and never clamps. `max_rows` is now plumbed
through `MoRELitModule` and `evaluate_split`, so the config key is real rather than decorative.

**Measured after the fix** (`scripts/probe_max_rows.py`, rel-event at `[12,12]`, 14 edge types):

| split | max rows/seed |
|---|---|
| train | 148 |
| **val** | **161–162** |
| **test** | **172** |

This is the part that matters. Training passed because train peaks at **148**, under the cap — the
crash always came at the first validation. But **test peaks at 172**. Raising `MAX_ROWS` to any value
inferable from the crash message (162, or even a padded 165) would have trained fine, validated fine,
and then died in **TEST eval at the very end of a full run**. There is no constant recoverable from
the failure itself that survives all three splits, because the splits are disjoint temporal eras with
different degree distributions.

**The general lesson:** a constant measured on one DB and asserted on all of them is a landmine, and
"measured max × 1.6" is not a safety margin when the quantity is schema-coupled. Worse, the obvious
repair — read the number off the assert and raise the constant — is itself a trap here, because the
split that overruns first is not the split that overruns most. Prefer fitting the axis to the batch
over raising the constant.

### 9.2 The MET offset assert, made diagnosable (4 failures)

`29029522_{8,34}` and `29029526_{16,33}` died on the known intermittent
`assert self.offset[0] == 0` inside a DataLoader worker — the un-normalisable branch of
`_patch_multiembedding_offset`. All four were also rel-event, and only occur with `num_workers>0`.

The layout is **still unknown**, and that is the point: `scripts/probe_met_offset.py` tried to
`print()` it, and worker stdout is discarded, so the probe could never have reported anything. The
fall-through now raises a `RuntimeError` carrying `(n_cols, k, T, values.shape, col_dims)` —
**exceptions cross the worker boundary, prints do not** — so the next occurrence will say what the
layout is. No repair is guessed: a wrong rebase would silently corrupt embeddings rather than crash.
Workaround meanwhile is unchanged: `--num-workers 0`.

### 9.3 Both grid arrays ran the WRONG experiment

Not a crash — worse, 96 jobs that completed and produced meaningless output. Reading the result
JSONs back:

| | intended | actually ran |
|---|---|---|
| `29029522` "qwen grid" | `--arch two_level`, `d_model{128,256} × n_blocks{2,4} × lr{3e-4,1e-3}` | `arch=rt`, `arch_grid()` configs 0–7 |
| `29029526` "harrier grid" | `--arch two_level --encoder harrier` | `arch=rt`, **`encoder=qwen`** |

Neither `--arch two_level` nor `--encoder harrier` reached the jobs. Evidence in every record:
`"arch": "rt"`, no `"phase"` key (two-level records carry one), no `"lr"` key, and
`n_heads=4, d_ff∈{256,512}, num_experts∈{4,8}` — the signature of `arch_grid()`, whose two-level
counterpart is `n_heads=8, d_ff=d_model×4, num_experts=4`.

Consequences:
- The array was sized `0-71` from `--list --arch two_level` (8 configs × 9 tasks), but indexed the
  **864-entry RT grid**, so it covered only RT configs 0–7 — all at `d_model=128, n_blocks=4`,
  varying only `num_experts` and `enc_channels`.
- `29029526` is therefore a **byte-for-byte duplicate** of `29029522`, and the harrier cache build
  `29029525` it depended on was consumed for nothing.
- **Nothing was learned about the two-level capacity/LR question the grid existed to answer.**

**The cause — and the first diagnosis here was WRONG.** This section originally concluded "the code
is not at fault … the submission simply omitted the flags." That was wrong, and believing it cost a
second wasted run: the grid was resubmitted on 2026-07-30 *with* `--arch two_level --encoder qwen`
explicitly on the `sbatch` line, and job `29030122_0` still wrote `"arch": "rt"`.

The real cause is one line in `run_gridsearch.py::main`:

```python
rec = run_index(args.index, seeds=args.seeds, epochs=args.epochs, num_workers=args.num_workers,
                seq_len=args.seq_len, max_fk=args.max_fk, out_dir=out_dir)
#               ^ arch / phase / encoder are PARSED and then dropped
```

`run_index` therefore fell back to its defaults `arch="rt", phase="full", encoder="qwen"` no matter
what was on the command line. What made it invisible is that **`--list`, in the same `main()`, DID
honour `args.arch`** — so `--list --arch two_level` correctly printed 72, the array was sized 72, and
every task then indexed the 864-entry RT grid. `--out-dir` was forwarded, which is why the results
landed in the right directory and looked plausible. `--encoder harrier` was dropped the same way,
which is why the "harrier grid" ran qwen and was a byte-for-byte duplicate.

`run_ablation.py` does **not** have this bug — it forwards `arch=args.arch` and resolves `--phase`
into `two_level=` — which is why the headline arrays genuinely ran two-level throughout.

Fixed by forwarding the three kwargs, plus `tests/test_runner_cli.py`, which asserts on the **whole**
forwarded kwarg set for both runners so the next knob added to a parser is covered by the same test.
A dropped kwarg is invisible at runtime — the job succeeds, it just answers a different question — so
it needs a test, not a code review.

**The habit that catches this class of thing in minutes** is to check the **first completed record**
against the intended config before letting an array run out; `arch`/`phase`/`encoder`/`lr` are in
every JSON precisely so this is a one-liner. It is what caught the second occurrence:

```bash
python -c "import json,glob;d=json.load(open(sorted(glob.glob('results/<dir>/*.json'))[0]));print({k:d.get(k) for k in ('arch','phase','encoder','d_model','n_blocks','lr')})"
```

### 9.4 What was resubmitted

`29030109` — headline array indices 18–26 (the nine rel-event runs), same flags as `29029490`.
**All 9 COMPLETED**, `results/two_level_full/` now 27/27: the `max_rows` fix holds on the real path.

The grids took two more attempts:

* `29030122`/`29030123` (2026-07-30 10:2x) — resubmitted whole rather than patching the 24 failed
  indices, since the 96 that "succeeded" were the wrong experiment. Cancelled by the user at 12:01
  because the array appeared to run one task at a time. That appearance was **cluster contention**,
  not a submission bug: all 12 h100s cluster-wide were allocated to another user's array, we held
  exactly one slot, and when task 0 released its GPU task 1 started the same second on the same node.
  `sacct` shows the throttle intact as `29030122_[2-71%8]`. Cancelling turned out to be lucky — the
  one record it did write proved §9.3 was still happening.
* `29030571` (qwen) → `29030572` (harrier, `afterany`) — submitted after the kwarg-forwarding fix,
  with the bogus RT record deleted from `results/tl_grid_qwen/` so the skip-guard could not honour it.

**`priority_gpu` is not usable by this account** — `sbatch --test-only --partition=priority_gpu`
returns "Invalid account or account/partition combination", and a `gpu,priority_gpu` list parks the
array in `(PartitionConfig)` indefinitely. `gpu` is the only GPU partition available, and its h100
pool is 12 GPUs shared cluster-wide.
