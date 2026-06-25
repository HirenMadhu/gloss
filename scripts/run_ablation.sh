#!/bin/bash
#SBATCH --job-name=gloss_abl
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Routing-signal ablation as a SLURM job ARRAY: one (dataset, task, signal, seed) config per task.
# Build the schema cache once first, then size the array to `run_ablation.py --list`:
#
#   .venv/bin/python scripts/build_schema_cache.py            # cache column-name embeddings (Qwen)
#   N=$(.venv/bin/python scripts/run_ablation.py --list --seeds 5)
#   sbatch --array=0-$((N-1))%8 scripts/run_ablation.sh --seeds 5
#   # when all tasks finish:
#   .venv/bin/python scripts/run_ablation.py --aggregate

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/run_ablation.py --index "${SLURM_ARRAY_TASK_ID}" \
    --encoder qwen --epochs 10 --batch-size 32 --num-workers 8 "$@"
echo "ABLATION_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
