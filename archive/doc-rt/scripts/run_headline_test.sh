#!/bin/bash
#SBATCH --job-name=gloss_headline_test
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

# DOC-RT HEADLINE GATE on the held-out RelBench TEST split, as a job ARRAY: 20 independent configs
# (5 seeds x {full,null,shuffled,name_only}), one H100 GPU each, up to 4 in parallel. Each config
# trains DOC-RT to convergence (identically to scripts/run_headline.sh), then scores the trained
# module on the TEST split via task.evaluate (RelBench masks test labels, so we cannot read them; the
# gate saves no checkpoints, so test eval reuses the in-memory module). Writes
# results/headline_test/<idx>.json with BOTH val and test metrics.
#
#   sbatch scripts/run_headline_test.sh
#   # when all tasks finish:
#   .venv/bin/python scripts/run_headline.py --aggregate --test
#
# Array size must equal `run_headline.py --list` for --seeds 5 (=20). Update --array if you change seeds.

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/run_headline.py --index "${SLURM_ARRAY_TASK_ID}" --seeds 5 \
    --encoder qwen --epochs 10 --batch-size 64 --num-workers 8 --test
echo "HEADLINE_TEST_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
