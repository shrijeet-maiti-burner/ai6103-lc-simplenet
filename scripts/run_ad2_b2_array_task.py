#!/usr/bin/env python3

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
CANDIDATES = [0.01, 0.1, 1.0]
SEED = 3
UPDATES = 256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tasks():
    tasks = []
    for category in CATEGORIES:
        tasks.append((category, "B1", 0.0, "shared"))
        for weight in CANDIDATES:
            tasks.append((category, "B2", weight, str(weight)))
    return tasks


def main():
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    tasks = build_tasks()

    if not 0 <= index < len(tasks):
        raise ValueError(f"Invalid array index: {index}")

    category, variant, weight, tag = tasks[index]

    study = json.loads((ROOT / "configs/ad2-full-study.json").read_text())
    selection = study["b2_selection"]

    if study["categories"] != CATEGORIES:
        raise ValueError("Category policy mismatch")
    if selection["candidates"] != CANDIDATES:
        raise ValueError("Lambda policy mismatch")
    if selection["updates_per_trial"] != UPDATES:
        raise ValueError("Update-count policy mismatch")
    if study["tuning_seed"] != SEED:
        raise ValueError("Tuning seed mismatch")
    if selection["validation_only"] is not True:
        raise ValueError("Selection must remain validation-only")

    manifest = ROOT / "references/ad2-normal-manifests" / f"{category}.json"
    manifest_hash = digest(manifest)

    output_root = ROOT / "results/ad2-b2-selection-array"
    run_dir = output_root / category / variant / tag / f"seed-{SEED}"
    metrics = run_dir / "normal-diagnostics.json"
    checkpoint = run_dir / "final.pt"

    if metrics.is_file() and checkpoint.is_file():
        print(f"Already complete; skipping: {run_dir}", flush=True)
        return

    if run_dir.exists():
        raise FileExistsError(f"Incomplete output exists; preserving it: {run_dir}")

    config = {
        "dataset": "mvtec_ad2",
        "category": category,
        "manifest_sha256": manifest_hash,
        "variant": variant,
        "lambda_score": weight,
        "seed": SEED,
        "batch_size": 8,
        "image_size": [256, 256],
        "ad_resize": None,
        "brightness": [0.8, 1.2],
        "gamma": [0.8, 1.2],
        "weights": "imagenet_v1",
    }

    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / (f"{category}-{variant}-{tag}-seed{SEED}.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    command = [
        sys.executable,
        "scripts/train_pilot.py",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--root",
        str(Path.home() / "datasets/mvtec_ad_2"),
        "--output",
        str(run_dir),
        "--steps",
        str(UPDATES),
        "--device",
        "cuda",
        "--threads",
        "4",
        "--gpu-memory-fraction",
        "0.8",
        "--normal-diagnostics",
    ]

    print(f"Starting task {index}: {category} {variant} {tag}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    if not metrics.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"Required outputs missing: {run_dir}")

    print(f"COMPLETED: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
