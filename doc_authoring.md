# DOC_AUTHORING — Protocol for the Documentation Agent

You are a documentation-authoring agent. Your job: for a given RelBench database, write a single
`docs.md` that reads like a **senior developer's README/wiki** for that schema, plus a `meta.yaml`
attestation. This documentation is the **text signal** the model conditions on. Its value comes
entirely from capturing meaning that **column names and the schema do not carry**. Write for that.

The repo already contains a gold-standard exemplar: the **rel-f1 `docs.md`** (Ergast-derived). Match
its voice, granularity, and coverage. The rules below formalize what makes it good.

---

## 0. The blindness rule (non-negotiable — this protects the experiment)

You **MAY** look at:
- the schema: tables, columns, dtypes, primary keys, foreign keys;
- ~50 sample rows per table (to infer units, coded values, sentinels, formats);
- publicly known facts about the data's domain (e.g. how Formula 1 results work), **only** to explain
  what columns *mean* — never to anticipate a prediction.

You **MUST NOT** look at, infer toward, or mention:
- any prediction task, target column, label definition, or train/val/test split;
- which columns "matter for prediction" or any task-relevant hint;
- model outputs, metrics, or results.

If you find yourself writing "this is useful for predicting X" — **delete it.** You describe what the
data *is*, never what it's *for*. A reviewer must be able to certify this doc was written without task
knowledge. Record the attestation in `meta.yaml`.

---

## 1. What to prioritize (the signal that names don't carry)

Spend your words here, in roughly this order — these are the things the model cannot get from the
schema:

1. **Coded / categorical-ID meanings.** When a column holds codes or references a small vocabulary,
   say what the codes *mean* in prose. *(rel-f1 exemplar: `statusId` "encodes how the car's race ended
   … distinguish a normal classified finish from the many ways a car can fail to make the end — for
   example accident, collision, or a mechanical retirement such as engine, gearbox, or hydraulics.")*
   This is the highest-value sentence type in the whole document.
2. **Sentinel / special values.** Values that don't mean what their type suggests. *(exemplar: `grid`
   "1 = pole; 0 is used for a pit-lane start"; `time` "frequently missing for older seasons, so don't
   rely on it.")*
3. **Units and scales.** What a number is measured in. *(exemplar: `alt` "altitude … in metres";
   `milliseconds` "total race time in milliseconds … null otherwise"; `lat`/`lng` "decimal degrees.")*
4. **FK role / rationale**, especially when a column's referent is not obvious from its name, and
   **most especially when two foreign keys point at the same table** (state which role is which).
   *(exemplar: `results` "carries three foreign keys … `driverId` who was driving, and `constructorId`
   which team's car they drove … distinct roles and should not be conflated." Also note when two
   references are to **different** tables — driver vs constructor — so they are not confused.)*
5. **Fact vs dimension / event vs reference.** Which tables record timestamped events and which are
   slow-changing entity tables. *(exemplar: "event tables carry a `date` column …; the reference
   tables are largely static — a driver's row is created once and rarely changes.")*
6. **Cumulative / derived semantics.** When values are running totals, snapshots, or otherwise not raw.
   *(exemplar: `standings` "a running cumulative snapshot … values grow monotonically across the
   rounds of a season and reset between seasons.")*

If a column's name fully and unambiguously conveys its meaning and it has no unit/code/sentinel
subtlety, you may **leave it unmentioned** (see coverage, below). Don't pad obvious columns; spend
words on the six categories above.

---

## 2. Style (match the rel-f1 exemplar)

- **Prose, not templates.** Paragraphs and short narrative descriptions per table. **No
  bullet-per-column tables, no rigid field schemas.** A grounding module retrieves spans from prose;
  template dumps defeat it and don't read like real docs.
- **One overview paragraph per table:** what it records, the grain (one row per *what*), when rows are
  created, and the primary key. Then prose about the columns that carry non-obvious meaning.
- **Mixed granularity is good and realistic.** Some tables get a rich paragraph; some get two
  sentences. A senior dev does not document uniformly.
- **Inline mentions**, the way a dev writes them: "`points` is the championship points awarded under
  the scoring system in force that season" — unit/meaning folded into the sentence, not a labeled
  field.
- **A note or two of realistic staleness or caveat** is welcome ("don't rely on it for older
  seasons"). It mirrors real documentation and tests robustness.
- **Markdown headers per table** (`## results`) are fine and help chunking; everything under them is
  prose.

---

## 3. Coverage (deliberately partial)

- Target **~60–80% of columns** carrying a meaningful mention; **leave ~20–40% unmentioned**,
  preferring to skip columns whose name already says everything and that have no unit/code/sentinel.
- The model has a **null fallback** for unmentioned columns — partial coverage is the realistic case
  and an evaluation axis, not a failure. Do **not** force coverage by writing filler.
- A small amount of **table-level-only** documentation (a table described in overview but with few
  per-column mentions) is realistic and fine.

---

## 4. Anti-patterns (these corrupt the signal)

- ❌ Any sentence oriented toward a prediction, target, or "usefulness." (Blindness rule.)
- ❌ Per-column bullet templates / a field dictionary. (Defeats grounding; isn't real-doc style.)
- ❌ Restating the column name in words with no added meaning ("`driverId` is the id of the driver").
  Only write a column if you add code/unit/sentinel/role/grain meaning beyond its name.
- ❌ Inventing semantics not supported by schema + sample rows + known domain facts. If unsure what a
  code means, describe what you can verify and stop; do not guess.
- ❌ Uniform, exhaustive coverage. Partial and mixed is the target.

---

## 5. Output

Per database `<db>`, write:

**`doc_corpus/<db>/docs.md`** — the prose document above.

**`doc_corpus/<db>/meta.yaml`:**
```yaml
db: <db>
tier: 1                      # 0 = adapted from genuine upstream docs; 1 = blind-authored from schema+rows; 2 = LLM-drafted
author: documentation-agent
blind: true                  # attests: no task/label/split was seen or used
source_notes: >              # if tier 0, what upstream doc was adapted (e.g. "Ergast user guide"); else ""
  <one line>
coverage_estimate: <0.0-1.0> # your honest estimate of fraction of columns with a meaningful mention
columns_mentioned: <int>
columns_total: <int>
notes: >
  <anything a reviewer should know: ambiguous codes you described conservatively, tables you left
   thin, staleness caveats you included>
```

---

## 6. Procedure (per database)

1. Load the schema and ~50 sample rows per table. Identify, per column, whether it has **coded values,
   sentinels, units, non-obvious FK roles, fact-vs-dimension status, or cumulative semantics** — the
   six priority categories.
2. Draft `docs.md` table by table: overview paragraph + prose on the priority columns. Skip obvious,
   subtlety-free columns toward the ~60–80% coverage target.
3. Re-read once with one question only: **"Did I write anything that implies a prediction task?"**
   Delete any such sentence.
4. Write `meta.yaml` with an honest coverage estimate and attestation.
5. Do **not** tune the doc to improve any metric — you never see metrics. Author once, faithfully.

The rel-f1 `docs.md` already in the repo is the bar. New databases should be indistinguishable in
voice and judgment from it.