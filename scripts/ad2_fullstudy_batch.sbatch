#!/bin/bash
#SBATCH --job-name=ad2-fullstudy
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ad2-fullstudy-%j.out
#SBATCH --error=logs/ad2-fullstudy-%j.err

set -euo pipefail

cd "$HOME/ai6103-lc-simplenet"

module load gcc/14.2.0
module load anaconda
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate ntu-gpu

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "=== AD2 FULL STUDY ==="
echo "JOB: $SLURM_JOB_ID"
echo "ALLOCATION: ${AD2_ALLOCATION:?Missing allocation}"
echo "NODE: $(hostname)"
echo "START: $(date)"

nvidia-smi

python -u scripts/ad2_fullstudy_scheduler.py

echo "=== ALLOCATION COMPLETED ==="
nvidia-smi
echo "FINISHED: $(date)"
