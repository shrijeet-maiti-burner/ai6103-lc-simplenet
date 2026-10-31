#!/bin/bash
#SBATCH --job-name=ad2-b2-l001
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ad2-b2-l001-%j.out
#SBATCH --open-mode=append

set -euo pipefail

module load gcc/14.2.0
module load anaconda
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate ntu-gpu

cd "$HOME/ai6103-lc-simplenet"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "B2 lambda=0.01 full experiment"
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

python -u scripts/ad2_b2_full_lambda001.py

echo "End: $(date)"
