#!/bin/bash
#SBATCH --job-name=gloss_testeval
#SBATCH --array=0-4%4
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

# HALOS rel-f1 driver-dnf TEST-set evaluation (5 seeds), headline config: full docs + generated geometry.
# Leaderboard-comparable -> vs GelGT 0.7608 +/- 0.0175. One seed per H100.
#   sbatch scripts/run_test_eval.sh
#   .venv/bin/python scripts/eval_test.py --aggregate

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; echo "seed ${SLURM_ARRAY_TASK_ID}"

python scripts/eval_test.py --index "${SLURM_ARRAY_TASK_ID}" --regime full \
    --epochs 10 --d-model 256 --n-layers 8 --batch-size 512 --num-workers 8
echo "TESTEVAL_${SLURM_ARRAY_TASK_ID}_DONE"
