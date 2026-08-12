# Shared environment for gloss jobs (source this from SLURM scripts and local runs).
#
# Redirects the HuggingFace / RelBench / graph / schema caches to a dedicated gloss subtree on
# scratch60 (the only writable scratch on this cluster; the PI scratch /nfs/roberts/scratch/pi_sk2433
# is not writable). A dedicated subtree keeps these separate from other workflows' caches (e.g. the
# MTEB cache at ~/scratch60/hub). Deliberately NOT added to ~/.bashrc.
#
#   source scripts/env.sh
export HF_HOME="${HF_HOME:-$HOME/scratch60/gloss/hf}"
export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$HOME/scratch60/gloss/relbench}"

# gloss's own caches: graph materialization + frozen schema (column-name) embeddings.
# gloss.utils.paths reads these (repo-relative fallback when unset, for hermetic tests).
export GLOSS_GRAPH_CACHE="${GLOSS_GRAPH_CACHE:-$HOME/scratch60/gloss/graph_cache}"
export GLOSS_SCHEMA_CACHE="${GLOSS_SCHEMA_CACHE:-$HOME/scratch60/gloss/schema_cache}"

# HF auth token (harrier is MIT / ungated, but this avoids anonymous download rate limits).
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

# Pretraining checkpoints + wandb run dirs. NOT home: a 520M-param AdamW state is ~7 GB per
# checkpoint and home's quota is ~125 GiB.
export GLOSS_CKPT_DIR="${GLOSS_CKPT_DIR:-$HOME/scratch60/gloss/ckpt}"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch60/gloss/wandb}"
