#!/bin/bash
#SBATCH --job-name=gloss_build_cache
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# One-time: download + embed the prose doc corpus with a (large) frozen encoder, ground every schema
# element, and cache d_c to data/doc_cache/<db>/emb_cache_<encoder>.pt. Needs a GPU big enough for the
# model in bf16 (the 27B 'harrier' is ~54 GB -> H100 80 GB; it would OOM the 46 GB A40). The corpus is
# tiny (~182 short texts), so once weights load the embedding pass is seconds.
#
#   sbatch scripts/build_doc_cache.sh                 # default ENCODER=harrier
#   ENCODER=harrier sbatch --export=ALL scripts/build_doc_cache.sh
#
# After it finishes, the headline gate arms read the cache and never load the model.

ENCODER="${ENCODER:-harrier}"

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "building doc cache: encoder=${ENCODER}"

python scripts/build_doc_cache.py --encoder "${ENCODER}"
echo "BUILD_CACHE_${ENCODER}_DONE"
