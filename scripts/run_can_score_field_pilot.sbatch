#!/bin/bash
# Validation-only can score-field pilot for the locked NTU TC2 stack.
# Create logs/ before submission. Slurm opens the log paths first.
# The allocated GPU is whatever MGPU-TC2 assigns to gpu:1. The pilot
# rejects anything other than an NVIDIA A40 before loading a head.
#SBATCH --job-name=can-score-field
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/can-score-field-pilot-%j.out
#SBATCH --error=logs/can-score-field-pilot-%j.err

set -Eeuo pipefail

for var in SCORE_FIELD_ROOT SCORE_FIELD_HEADS SCORE_FIELD_DATA SCORE_FIELD_OUT SCORE_FIELD_WEIGHTS
do
    if [[ -z "${!var:-}" ]]; then
        echo "Missing ${var}" >&2
        exit 1
    fi
done

SCORE_FIELD_PROVENANCE="${SCORE_FIELD_PROVENANCE:-$SCORE_FIELD_ROOT/provenance/can-pilot-code.json}"
SCORE_FIELD_PROVENANCE_SHA256="${SCORE_FIELD_PROVENANCE_SHA256:-$SCORE_FIELD_ROOT/provenance/can-pilot-code.sha256}"
STUDY_LOCK_SHA256="3d3b9dd4995622d26fb3077f8000255f974a03c715ebe764a64eeedac3e53aa2"

[[ -d "$SCORE_FIELD_ROOT" ]] || { echo "SCORE_FIELD_ROOT is not a directory" >&2; exit 1; }
[[ -d "$SCORE_FIELD_HEADS" ]] || { echo "SCORE_FIELD_HEADS is not a directory" >&2; exit 1; }
[[ -d "$SCORE_FIELD_DATA" ]] || { echo "SCORE_FIELD_DATA is not a directory" >&2; exit 1; }
[[ -f "$SCORE_FIELD_WEIGHTS" ]] || { echo "SCORE_FIELD_WEIGHTS is not a file" >&2; exit 1; }
[[ -f "$SCORE_FIELD_PROVENANCE" ]] || { echo "Provenance manifest is missing" >&2; exit 1; }
[[ -f "$SCORE_FIELD_PROVENANCE_SHA256" ]] || { echo "Provenance sidecar is missing" >&2; exit 1; }
if [[ -e "$SCORE_FIELD_OUT" && ! -d "$SCORE_FIELD_OUT" ]]; then
    echo "SCORE_FIELD_OUT exists and is not a directory" >&2
    exit 1
fi
if [[ -d "$SCORE_FIELD_OUT" ]] && [[ -n "$(ls -A "$SCORE_FIELD_OUT")" ]]; then
    echo "SCORE_FIELD_OUT is not empty" >&2
    exit 1
fi

cd "$SCORE_FIELD_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${SCORE_FIELD_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

uv_line="$(uv --version)"
case "$uv_line" in
    "uv 0.13.0 "*|"uv 0.13.0") ;;
    *)
        echo "uv 0.13.0 is required; found: ${uv_line}" >&2
        exit 1
        ;;
esac

python3 - "$SCORE_FIELD_PROVENANCE" "$SCORE_FIELD_PROVENANCE_SHA256" "$SCORE_FIELD_ROOT/uv.lock" "$STUDY_LOCK_SHA256" << 'PY'
import hashlib
import sys
from pathlib import Path

manifest, sidecar, lock, expected_lock = sys.argv[1:]
manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
recorded = Path(sidecar).read_text(encoding="utf-8").strip().split()
if not recorded or recorded[0] != manifest_hash:
    raise SystemExit("Provenance manifest hash does not match the sidecar")
if len(recorded) > 1 and Path(recorded[1]).name != Path(manifest).name:
    raise SystemExit("Provenance sidecar names a different manifest")
lock_hash = hashlib.sha256(Path(lock).read_bytes()).hexdigest()
if lock_hash != expected_lock:
    raise SystemExit("uv.lock does not match the formal can study lock")
print(f"provenance {manifest_hash}")
print(f"uv.lock {lock_hash}")
PY

if [[ ! -x "$SCORE_FIELD_ROOT/.venv/bin/python" ]]; then
    echo "No project-local .venv is present." >&2
    echo "This job does not install or upgrade packages and does not use the network." >&2
    echo "On a login node with network, from SCORE_FIELD_ROOT, create only this project's environment:" >&2
    echo "  uv sync --frozen --extra cuda" >&2
    echo "Do not activate or modify the shared ntu-gpu environment." >&2
    exit 1
fi

mkdir -p logs "$SCORE_FIELD_OUT"

# --frozen uses uv.lock. --offline and --no-sync refuse downloads and upgrades.
if ! uv run --frozen --offline --no-sync --extra cuda \
    python -c 'import platform; raise SystemExit(0 if platform.python_version()=="3.12.15" else 1)'
then
    echo "The locked CPython 3.12.15 CUDA environment is not available offline on this node." >&2
    echo "This job does not install or upgrade packages and does not use the network." >&2
    exit 1
fi

uv run --frozen --offline --no-sync --extra cuda python -u scripts/run_can_score_field_pilot.py \
    --source-root "$SCORE_FIELD_ROOT" \
    --provenance "$SCORE_FIELD_PROVENANCE" \
    --provenance-sha256 "$SCORE_FIELD_PROVENANCE_SHA256" \
    --heads-root "$SCORE_FIELD_HEADS" \
    --data-root "$SCORE_FIELD_DATA" \
    --out-dir "$SCORE_FIELD_OUT" \
    --weights "$SCORE_FIELD_WEIGHTS" \
    --device cuda \
    --threads 3
