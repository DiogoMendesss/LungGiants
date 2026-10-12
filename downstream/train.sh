#!/bin/bash
#
#SBATCH --partition=gpu_min32gb       # GPU partition. Adjust to your cluster naming.
#SBATCH --qos=gpu_min32gb              # QoS level. Must usually match the partition.
#SBATCH --job-name=train_real100 # Job name
#SBATCH --output=slurm_out/train/slurm_%x.%j.out
#SBATCH --error=slurm_out/train/slurm_%x.%j.err

# Commands / scripts to run
python -u train.py \
    --experiment configs/experiments/real_100.yaml \
    "$@"

