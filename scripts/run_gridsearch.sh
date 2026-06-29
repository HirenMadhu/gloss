#!/bin/bash
#SBATCH --job-name=gloss_grid
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Architecture grid search (signature arm, harrier cache) as a SLURM job ARRAY: one (config, task, seed)
# per array index. Caches (graph + harrier schema) must already exist for rel-f1 / rel-trial / rel-event.
#
#   N=$(.venv/bin/python scripts/run_gridsearch.py --list)        # = 96 configs x 9 tasks x 3 seeds = 2592
#   sbatch --array=0-$((N-1))%16 scripts/run_gridsearch.sh
#   # when done:
#   .venv/bin/python scripts/run_gridsearch.py --aggregate

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "grid task ${SLURM_ARRAY_TASK_ID}"

python scripts/run_gridsearch.py --index "${SLURM_ARRAY_TASK_ID}" \
    --epochs 30 --seeds 3 --num-workers 8 "$@"
rc=$?
echo "GRID_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
