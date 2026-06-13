#!/bin/bash
#SBATCH --job-name=gloss_testeval
#SBATCH --array=0-4%5
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

# RelBench TEST-set evaluation for one (DATASET, TASK, REGIME), one seed per array task.
# Parameterized via --export; submit once per task:
#   sbatch --export=ALL,DATASET=rel-f1,TASK=driver-top3,REGIME=full   scripts/run_test_eval.sh
#   sbatch --export=ALL,DATASET=rel-event,TASK=user-repeat,REGIME=null scripts/run_test_eval.sh
#   sbatch --export=ALL,DATASET=rel-trial,TASK=study-outcome,REGIME=null scripts/run_test_eval.sh
# then: .venv/bin/python scripts/eval_test.py --aggregate

mkdir -p ./logs/slurm
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
DATASET=${DATASET:-rel-f1}; TASK=${TASK:-driver-dnf}; REGIME=${REGIME:-full}
date; hostname; echo "seed ${SLURM_ARRAY_TASK_ID} | ${DATASET}/${TASK} regime=${REGIME}"

python scripts/eval_test.py --index "${SLURM_ARRAY_TASK_ID}" \
    --dataset "${DATASET}" --task "${TASK}" --regime "${REGIME}" \
    --epochs 10 --d-model 256 --n-layers 8 --batch-size 512 --num-workers 8
echo "TESTEVAL_${DATASET}_${TASK}_${SLURM_ARRAY_TASK_ID}_DONE"
