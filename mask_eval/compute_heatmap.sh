#!/bin/bash
#
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores  
#SBATCH --job-name=heatmap
#SBATCH --output=slurm_out/slurm_%x.%j.out
#SBATCH --error=slurm_out/slurm_%x.%j.err

set -euo pipefail

MASKS_DIR="/nas-ctm01/datasets/public/LungNodule-nifti/preprocessed/MAISI/results_synth_maskct/masks_132_without_tumor"

python -u mask_eval/compute_heatmap.py \
        --masks-dir $MASKS_DIR \
