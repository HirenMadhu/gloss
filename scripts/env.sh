# Shared environment for gloss jobs (source this from SLURM scripts and local runs).
#
# Redirects the HuggingFace and RelBench caches to scratch60 (large, fast, 60-day scratch).
# Deliberately NOT added to ~/.bashrc so we don't disturb other workflows that use a
# different HF cache (e.g. the MTEB cache at ~/scratch60/hub).
#
#   source scripts/env.sh
export HF_HOME="${HF_HOME:-$HOME/scratch60/hf}"
export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$HOME/scratch60/relbench}"

# HF auth token (harrier is MIT / ungated, but this avoids anonymous download rate limits).
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
