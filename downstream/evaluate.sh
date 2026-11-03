#!/bin/bash
#
#SBATCH --partition=gpu_min32gb       # GPU partition. Adjust to your cluster naming.
#SBATCH --qos=gpu_min32gb              # QoS level. Must usually match the partition.
#SBATCH --job-name=evaluate_nodules   # Job name
#SBATCH --output=slurm_out/evaluate/slurm_%x.%j.out
#SBATCH --error=slurm_out/evaluate/slurm_%x.%j.err

# Commands / scripts to run (edit model and output paths before submitting).
python -u evaluate.py \
    --model-path /path/to/model.ckpt \
    --set both \
    --output-path reports/evaluation \
    --device cuda \
    --precision 16-mixed \
    "$@"
