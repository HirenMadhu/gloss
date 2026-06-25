#!/bin/bash
#SBATCH --job-name=gloss_headline_harrier
#SBATCH --array=0-19%4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# DOC-RT HEADLINE GATE using the 'harrier' (microsoft/harrier-oss-v1-27b, d_text=5376) doc embeddings,
# as a job ARRAY: 20 configs (5 seeds x {full,null,shuffled,name_only}), one H100 each, up to 4 parallel.
# Identical training to scripts/run_headline_test.sh; only the cached d_c differs. Reads the prebuilt
# data/doc_cache/rel-f1/emb_cache_harrier.pt (NO model load in these arms). Records BOTH val and test
# metrics to results/headline_test_harrier/<idx>.json. Submit with a dependency on the cache build:
#
#   BUILD=$(sbatch --parsable scripts/build_doc_cache.sh)
#   sbatch --dependency=afterok:$BUILD scripts/run_headline_harrier.sh
#   # when all tasks finish:
#   .venv/bin/python scripts/run_headline.py --aggregate --test --encoder harrier

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/run_headline.py --index "${SLURM_ARRAY_TASK_ID}" --seeds 5 \
    --encoder harrier --epochs 10 --batch-size 64 --num-workers 8 --test
echo "HEADLINE_HARRIER_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
