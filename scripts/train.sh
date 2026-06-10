#!/bin/bash

#SBATCH --job-name=gloss_train
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

date
hostname
pwd

# DDP rendezvous (single node). Bump nproc_per_node + --gpus to scale.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29004
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py experiment.wandb.name=run1
