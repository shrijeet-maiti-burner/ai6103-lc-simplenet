#!/usr/bin/env python3
"""Normal-only B2 lambda selection for the frozen MVTec AD 2 study."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]
CANDIDATES = [0.01, 0.1, 1.0]
CORNERS = [[0.8, 0.8], [0.8, 1.2], [1.2, 0.8], [1.2, 1.2]]
SEED = 3
UPDATES = 256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def validate_metrics(metrics, expected_hash, category, variant, weight):
    if metrics.get("manifest_sha256") != expected_hash:
        raise ValueError(f"{category}/{variant}/{weight}: manifest hash mismatch")

    if metrics.get("images", 0) < 1:
        raise ValueError(f"{category}/{variant}/{weight}: empty validation set")

    if metrics.get("lighting_corners") != CORNERS:
        raise ValueError(
            f"{category}/{variant}/{weight}: lighting corners differ: "
            f"{metrics.get('lighting_corners')}"
        )

    for key in ("pair_mse", "patch_score_std", "synthetic_hinge"):
        value = metrics.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{category}/{variant}/{weight}: invalid {key}")

    if metrics["pair_mse"] < 0 or metrics["patch_score_std"] < 0:
        raise ValueError(f"{category}/{variant}/{weight}: negative metric")


def select_weight(baselines, trials):
    evaluations = []

    for weight in CANDIDATES:
        ratios = []
        eligible = True
        category_checks = {}

        for category in CATEGORIES:
            base = baselines[category]
            current = trials[category][weight]

            if base["patch_score_std"] <= 1e-6:
                eligible = False
                category_checks[category] = {
                    "eligible": False,
                    "reason": "B1 patch-score standard deviation is too small",
                }
                continue

            std_ratio = current["patch_score_std"] / base["patch_score_std"]
            hinge_ratio = current["synthetic_hinge"] / max(base["synthetic_hinge"], 1e-12)
            mse_ratio = current["pair_mse"] / max(base["pair_mse"], 1e-12)

            category_ok = (
                std_ratio >= 0.5 and current["synthetic_hinge"] <= 1.25 * base["synthetic_hinge"]
            )
            eligible = eligible and category_ok
            ratios.append(mse_ratio)

            category_checks[category] = {
                "eligible": category_ok,
                "relative_pair_mse": mse_ratio,
                "patch_score_std_ratio": std_ratio,
                "synthetic_hinge_ratio": hinge_ratio,
            }

        evaluations.append(
            {
                "lambda_score": weight,
                "eligible": bool(eligible),
                "category_macro_relative_pair_mse": (
                    sum(ratios) / len(ratios) if len(ratios) == len(CATEGORIES) else None
                ),
                "category_checks": category_checks,
            }
        )

    feasible = [
        row
        for row in evaluations
        if row["eligible"] and row["category_macro_relative_pair_mse"] is not None
    ]

    if not feasible:
        raise RuntimeError(
            "STOP: no B2 candidate satisfies all frozen safeguards. "
            "Do not relax thresholds or inspect test data."
        )

    best = min(row["category_macro_relative_pair_mse"] for row in feasible)
    chosen = min(
        row["lambda_score"]
        for row in feasible
        if row["category_macro_relative_pair_mse"] <= best * 1.01 + 1e-12
    )

    return {
        "status": "selected",
        "lambda_score": chosen,
        "selection_seed": SEED,
        "updates_per_run": UPDATES,
        "categories": CATEGORIES,
        "candidates": evaluations,
        "rule": (
            "Smallest positive eligible lambda within 1% of the best "
            "category-macro relative B2/B1 pair-MSE ratio"
        ),
        "scope": "normal-validation selection only; no defect sensitivity established",
        "test_data_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; request a GPU allocation first")

    if args.threads < 1 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("Invalid thread count or GPU memory fraction")

    policy = read_json(ROOT / "configs/ad2-study-policy.json")
    study = read_json(ROOT / "configs/ad2-full-study.json")

    # Cross-check the full-study configuration against the frozen policy.
    expected_categories = policy["selected_categories"]
    if expected_categories != CATEGORIES or study["categories"] != CATEGORIES:
        raise ValueError("Category scope differs from the frozen eight-category policy")

    for key in (
        "image_size",
        "batch_size",
        "main_seeds",
        "variants",
        "source_passes",
        "brightness",
        "gamma",
    ):
        if policy[key] != study[key]:
            raise ValueError(f"Policy/config mismatch for {key}: {policy[key]} != {study[key]}")

    if study["tuning_seed"] != SEED:
        raise ValueError("Tuning seed differs from the frozen selection seed")

    if study["image_size"] != [256, 256] or study["batch_size"] != 8:
        raise ValueError("Unexpected selection image geometry or batch size")

    if study["variants"] != ["B0", "B1", "B2"]:
        raise ValueError("Unexpected variant scope")

    if study["test_data_used_for_selection"] is not False:
        raise ValueError("Test data must remain excluded from selection")

    selection = study["b2_selection"]
    expected_selection = {
        "updates_per_trial": UPDATES,
        "candidates": CANDIDATES,
        "shared_b1_control_per_category": True,
        "validation_only": True,
        "brightness_gamma_corners": CORNERS,
        "minimum_patch_std_ratio": 0.5,
        "maximum_synthetic_hinge_ratio": 1.25,
        "tie_tolerance": 0.01,
        "no_eligible_candidate": "stop",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"B2 policy mismatch for {key}: {selection.get(key)!r} != {expected!r}"
            )

    manifest_dir = ROOT / "references/ad2-normal-manifests"
    output_root = ROOT / "results/ad2-b2-selection-array"
    config_dir = output_root / "configs"

    jobs = []
    for category in CATEGORIES:
        manifest = manifest_dir / f"{category}.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        manifest_hash = digest(manifest)
        jobs.append((category, "B1", 0.0, manifest, manifest_hash))
        for weight in CANDIDATES:
            jobs.append((category, "B2", weight, manifest, manifest_hash))

    print(f"Planned runs: {len(jobs)}")
    print("8 shared B1 controls + 24 B2 candidate runs")
    print(f"Updates per run: {UPDATES}")
    print(f"Total optimizer updates: {len(jobs) * UPDATES}")

    if args.dry_run:
        for category, variant, weight, *_ in jobs:
            print(category, variant, weight)
        return

    for category, variant, weight, manifest, manifest_hash in jobs:
        weight_tag = "shared" if variant == "B1" else str(weight)
        run_dir = output_root / category / variant / weight_tag / f"seed-{SEED}"
        metrics_path = run_dir / "normal-diagnostics.json"

        if metrics_path.is_file() and (run_dir / "final.pt").is_file():
            print(f"Existing completed run: {category} {variant} {weight_tag}")
            continue

        if run_dir.exists():
            raise FileExistsError(
                f"Incomplete run directory exists: {run_dir}. "
                "Preserve it and investigate before retrying."
            )

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

        config_path = config_dir / f"{category}-{variant}-{weight_tag}-seed{SEED}.json"
        write_json(config_path, config)

        command = [
            sys.executable,
            "scripts/train_pilot.py",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--root",
            str(args.root),
            "--output",
            str(run_dir),
            "--steps",
            str(UPDATES),
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--gpu-memory-fraction",
            str(args.gpu_memory_fraction),
            "--normal-diagnostics",
        ]

        print("\nRUN:", category, variant, weight_tag, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

        if not metrics_path.is_file() or not (run_dir / "final.pt").is_file():
            raise RuntimeError(f"Run did not produce required outputs: {run_dir}")

    baselines = {}
    trials = {category: {} for category in CATEGORIES}

    for category in CATEGORIES:
        manifest_hash = digest(manifest_dir / f"{category}.json")

        base_dir = output_root / category / "B1" / "shared" / f"seed-{SEED}"
        base = read_json(base_dir / "normal-diagnostics.json")
        validate_metrics(base, manifest_hash, category, "B1", "shared")
        baselines[category] = base

        for weight in CANDIDATES:
            run_dir = output_root / category / "B2" / str(weight) / f"seed-{SEED}"
            metrics = read_json(run_dir / "normal-diagnostics.json")
            validate_metrics(metrics, manifest_hash, category, "B2", weight)
            trials[category][weight] = metrics

    result = select_weight(baselines, trials)
    result["baseline_runs"] = {
        category: str(output_root / category / "B1" / "shared" / f"seed-{SEED}")
        for category in CATEGORIES
    }
    result["candidate_runs"] = {
        category: {
            str(weight): str(output_root / category / "B2" / str(weight) / f"seed-{SEED}")
            for weight in CANDIDATES
        }
        for category in CATEGORIES
    }

    report = output_root / "b2-selection.json"
    write_json(report, result)
    print("\nB2 SELECTION COMPLETE")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {report}")


if __name__ == "__main__":
    main()
