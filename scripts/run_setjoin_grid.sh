#!/bin/bash
#SBATCH --job-name=gloss_sj_grid
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# SetJoin backbone sweep (S10) as a SLURM job ARRAY: one (config, task, seed) cell per task —
# MILGRAM-native (partition=gpu, h100:1, hard 8-GPU QOS cap => submit with %8).
#
# BEFORE submitting: verify the rel-event scratch symlinks exist (recap.md §2) —
#   ~/scratch60/gloss/relbench/rel-event, .../graph_cache/rel-event, .../schema_cache/rel-event
#
#   N=$(.venv/bin/python scripts/run_setjoin_grid.py --list)     # 486
#   sbatch --array=0-$((N-1))%8 scripts/run_setjoin_grid.sh
#   # when all tasks finish:
#   .venv/bin/python scripts/run_setjoin_grid.py --aggregate

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}  HF_HOME=$HF_HOME RELBENCH_CACHE_DIR=$RELBENCH_CACHE_DIR"

python scripts/run_setjoin_grid.py --index "${SLURM_ARRAY_TASK_ID}" --num-workers 8 "$@"
rc=$?
echo "SJ_GRID_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
