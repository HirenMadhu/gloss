# gloss — HALOS

**HALOS** (*Heterogeneous Attention, Language Of Schema*) — a **measurement paper**:
*"names lie, meaning transfers."* Across heterogeneous relational databases, column *names* are a brittle
proxy; the transferable signal is documented column *meaning* (units, null semantics, coded-value
dictionaries, FK-role descriptions). We **prove** the model uses meaning, not leaked names/labels.

Two headline contributions (the temporal kernel C2 is deferred to Paper #2):
- **C1 — Structured DocCards**: per-column documentation as a frozen-LM modality, FiLM-fused into cells;
  also fixes RT's dual-foreign-key ambiguity via distinct FK-role ids.
- **C3 — Documentation Sufficiency Audit (DSA)**: a model-agnostic information-theoretic + faithfulness
  audit (`Î(Y;Doc|Values,Structure)` + placebo + blind-authoring + Shapley) — *the product and the moat*.

See [idea.md](idea.md) (rationale) and [implementation.md](implementation.md) (build spec) — both
normative — and [PROGRESS.md](PROGRESS.md) for the phase-by-phase log.

## Layout (`gloss/`)
```
data/   relbench_graph (leakage-safe temporal graph + sampler), doccards, doccard_authoring,
        text_cache (frozen Qwen3-Embedding-4B), synthetic (planted-truth generator), collate
proxy/  embed_probe                         # Phase 2: GATE 1 — the week-one proxy test
model/  tokenizer fusion time_simple biases attention halos heads   # Phase 3 (stubs until GATE 1)
audit/  cmi controls shapley faithfulness runner readback           # Phase 5 — THE PRODUCT
train/  loop finetune pretrain losses       # Phase 4 (Lightning)
eval/   gradient nameshuffle metrics         # Phase 6 (the headline map)
utils/  seeding config logging flops
ext/    time_scaleequiv                      # DEFERRED — Paper #2, do not build
```

## Setup (Milgram / el8, CUDA 12.8)
```bash
uv venv --python 3.12 .venv
bash scripts/setup_env.sh            # staged install: torch 2.8.0+cu128 + PyG + relbench + Qwen + harness
sbatch scripts/build_flash_attn.sh   # optional: flash-attn from source (sm_86;sm_90); off the gate path
```
Large caches go to scratch: `export HF_HOME=~/scratch60/hf GLOSS_SCRATCH=~/scratch60`.

## Run
```bash
python scripts/run_finetune.py --dry-run                 # Phase 0 DoD: sample a batch, print shapes
python scripts/build_text_cache.py --dataset synthetic --encoder qwen --regime all   # Phase 1
python scripts/run_proxy_gate.py                         # Phase 2: GATE 1 (the first real result)
pytest                                                   # Phase 0–1 green; Phase 3/5 skipped
```

## Working agreement
Measurement-first (not SOTA); **proxy before transformer** (stop at GATE 1); the audit is the deliverable;
small & reproducible (≤~30M params, frozen cached text encoder, global seed). Don't build the C2 kernel.
