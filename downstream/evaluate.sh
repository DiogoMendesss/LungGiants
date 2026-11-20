#!/bin/bash
#
#SBATCH --partition=debug_8gb     # GPU partition. Adjust to your cluster naming.
#SBATCH --qos=debug_8gb              # QoS level. Must usually match the partition.
#SBATCH --job-name=evaluate_nodules   # Job name
#SBATCH --output=slurm_out/evaluate/slurm_%x.%j.out
#SBATCH --error=slurm_out/evaluate/slurm_%x.%j.err

# Commands / scripts to run (edit model and output paths before submitting).
python -u evaluate.py \
    --model-path runs_lung+nodule/real_100/20261020_190006_782295/checkpoints/epoch=035-nodule_dice=0.4985.ckpt \
    --set test \
    --output-path reports/real100_plot \
    --device cuda \
    --precision 16-mixed \
    "$@"
