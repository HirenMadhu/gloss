#!/bin/bash
#SBATCH --job-name=gloss_prep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# One-time prep before the ablation array: download rel-stack, materialize task tables, populate the
# graph cache, and build the Qwen schema cache for rel-f1 / rel-stack / rel-trial.
#   sbatch scripts/prep.sh

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
date; hostname

python scripts/prep_data.py --encoder qwen "$@"
echo "PREP_DONE"
