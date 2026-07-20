#!/bin/bash
#SBATCH --job-name=gloss_chain_gates
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Fires after the S10 sweep array drains (submit with --dependency=afterany:<array job id>):
# aggregates the sweep, picks the adopted <=30M backbone, and sbatches the three gate arrays.

cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate
source scripts/env.sh
date; hostname
python scripts/chain_gates.py
