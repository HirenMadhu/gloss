#!/bin/bash
#SBATCH --job-name=gloss_ablfus
#SBATCH --array=0-14%4
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

# Doc cross-attention fusion ablation (15 configs = 3 seeds x {full:film,feature,geometry,both} + null:film),
# one H100 each, 4 in parallel. Array size = `ablation_fusion.py --list` for --seeds 3 (=15).
#   sbatch scripts/run_ablation.sh
#   .venv/bin/python scripts/ablation_fusion.py --aggregate   # when done

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
date; hostname; echo "array task ${SLURM_ARRAY_TASK_ID}"

python scripts/ablation_fusion.py --index "${SLURM_ARRAY_TASK_ID}" --seeds 3 \
    --epochs 10 --d-model 256 --n-layers 8 --batch-size 512 --num-workers 8
echo "ABL_TASK_${SLURM_ARRAY_TASK_ID}_DONE"
