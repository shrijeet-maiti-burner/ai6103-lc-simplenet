#!/bin/bash
#SBATCH --job-name=ad2-fabric
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/ad2_fabric_batch_%j.out
#SBATCH --error=logs/ad2_fabric_batch_%j.err

set -Eeuo pipefail
cd "$HOME/ai6103-lc-simplenet"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p logs results

for variant in B0 B1 B2; do
    lower=$(printf '%s' "$variant" | tr '[:upper:]' '[:lower:]')
    config="results/ad2_fabric_${lower}_config.json"
    output="results/ad2_fabric_${lower}_5000steps_recovery"

    [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }

    if [[ -f "$output/final.pt" ]]; then
        echo "$variant already completed; skipping."
        continue
    fi

    resume_args=()
    latest_checkpoint=""

    if [[ -d "$output" ]]; then
        latest_checkpoint=$(find "$output" -maxdepth 1 -type f \
            -name 'checkpoint-*.pt' -printf '%f\n' |
            sort -V | tail -n 1 || true)
    fi

    if [[ -n "$latest_checkpoint" ]]; then
        resume_args=(--resume "$output/$latest_checkpoint")
        echo "Resuming $variant from $latest_checkpoint"
    elif [[ -d "$output" ]]; then
        echo "$variant: $output exists without a periodic checkpoint; refusing a fresh start." >&2
        exit 1
    else
        echo "Starting $variant from step 0"
    fi

    uv run --extra cuda python scripts/train_pilot.py \
        --config "$config" \
        --manifest data/manifests/ad2-fabric.jsonl \
        --root data/mvtec_ad2 \
        --output "$output" \
        --steps 5000 \
        --device cuda \
        --threads 4 \
        --target-fpr 0.05 \
        --checkpoint-every 250 \
        "${resume_args[@]}" \
        2>&1 | tee -a "logs/ad2_fabric_${lower}_5000steps_recovery.log"

    echo "$variant finished at $(date -Is)"
done
