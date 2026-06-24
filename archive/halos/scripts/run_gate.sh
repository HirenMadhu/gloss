#!/bin/bash
#SBATCH --job-name=gloss_gate
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

# GATE 1 (H1 + H2) as a job ARRAY: 20 independent configs (5 seeds x {full,shuffled_spans,null}-generated
# + full-free_learned), one H100 GPU each, up to 4 in parallel. Each config trains HALOS to convergence
# (10 epochs, full batches) and writes results/gate/<idx>.json. (Multi-GPU via parallel array tasks; each
# run is single-GPU — the runs are independent, so DDP would add nothing.)
#
#   sbatch scripts/run_gate.sh
#   # when all tasks finish:
#   .venv/bin/python scripts/gate_run.py --aggregate
#
# Array size must equal `gate_run.py --list` for --seeds 5 (=20). Update --array if you change seeds.

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/gate_run.py --index "${SLURM_ARRAY_TASK_ID}" --seeds 5 \
    --epochs 10 --d-model 256 --n-layers 8 --batch-size 512 --num-workers 8
echo "GATE_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
