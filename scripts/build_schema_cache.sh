#!/bin/bash
#SBATCH --job-name=gloss_schema_cache
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Precompute the frozen column-name embeddings (Qwen) for every dataset, once, before the ablation array.
#   sbatch scripts/build_schema_cache.sh

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh          # REQUIRED: without it GLOSS_SCHEMA_CACHE falls back to the
                               # repo-relative data/schema_cache/, which is NOT where training
                               # jobs look (they read $HOME/scratch60/gloss/schema_cache), and
                               # HF_HOME falls back to $HOME/.cache — downloading an 8 GB model
                               # onto the home filesystem instead of scratch. Every other SLURM
                               # script here (prep.sh, run_ablation.sh, run_gridsearch.sh) sources
                               # it; this one did not, so its output landed where nothing reads it.
date; hostname; echo "HF_HOME=$HF_HOME GLOSS_SCHEMA_CACHE=$GLOSS_SCHEMA_CACHE"

python scripts/build_schema_cache.py --encoder qwen "$@"
echo "SCHEMA_CACHE_DONE"
