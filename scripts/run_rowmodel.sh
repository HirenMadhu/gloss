#!/bin/bash
#SBATCH --job-name=gloss_rowmodel
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# RowModel (unified hierarchical wide-row) gate as a SLURM job ARRAY: one (dataset, task, seed)
# cell per task — MILGRAM-native (partition=gpu, h100:1, hard 8-GPU QOS cap => submit with %8).
#
# The cell-level MoE runs over the full [B, M_rows, C] grid (~48x the two-stream token load), so this
# passes --checkpoint-cells (gradient-checkpoint the cell layers; bf16 AMP is on for rowmodel) and a
# conservative --batch-size 64 for first-attempt headroom on the H100 (the runner's OOM-retry is the
# backstop, but a full-restart on a late OOM risks the 24h wall clock).
#
# BEFORE submitting: verify the rel-event scratch symlinks exist (recap.md §2) —
#   ~/scratch60/gloss/relbench/rel-event, .../graph_cache/rel-event, .../schema_cache/rel-event
#
#   N=$(.venv/bin/python scripts/run_setjoin.py --list)        # 27 (9 tasks x 3 seeds)
#   sbatch --array=0-$((N-1))%8 scripts/run_rowmodel.sh --out-dir results/rowmodel
#   # when all tasks finish:
#   .venv/bin/python scripts/run_setjoin.py --aggregate --model rowmodel --out-dir results/rowmodel
#   .venv/bin/python scripts/run_setjoin.py --compare  --out-dir results/rowmodel

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}  HF_HOME=$HF_HOME RELBENCH_CACHE_DIR=$RELBENCH_CACHE_DIR"

python scripts/run_setjoin.py --index "${SLURM_ARRAY_TASK_ID}" \
    --model rowmodel --d-model 256 --n-cell-layers 2 --n-row-layers 2 --d-ff 1024 \
    --route-on signature --checkpoint-cells --batch-size 64 \
    --encoder harrier --num-workers 8 "$@"
rc=$?
echo "ROWMODEL_TASK_${SLURM_ARRAY_TASK_ID}_DONE (rc=$rc)"
exit $rc
