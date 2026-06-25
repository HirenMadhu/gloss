#!/bin/bash
#SBATCH --job-name=gloss_schema_cache
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Precompute the frozen column-name embeddings (Qwen) for every dataset, once, before the ablation array.
#   sbatch scripts/build_schema_cache.sh

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
date; hostname

python scripts/build_schema_cache.py --encoder qwen "$@"
echo "SCHEMA_CACHE_DONE"
