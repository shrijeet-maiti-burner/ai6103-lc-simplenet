#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "results/ad2-full-study/ad2-task-queue.json"
STUDY = ROOT / "configs/ad2-full-study.json"
TRAINER = ROOT / "scripts/train_pilot.py"

OUT_ROOT = ROOT / "results/ad2-b2-full-lambda001"
CONFIG_DIR = OUT_ROOT / "configs"
LOG_DIR = ROOT / "logs"

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
SEEDS = [0, 1, 2]
LAMBDA = 0.01

EXPECTED_STEPS = {
    "can": 4160,
    "fabric": 3920,
    "fruit_jelly": 2640,
    "rice": 3200,
    "sheet_metal": 1440,
    "vial": 2960,
    "wallplugs": 2960,
    "walnuts": 4320,
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(out, steps):
    required = [
        out / "final.pt",
        out / "run.json",
        out / "result.json",
        out / "validation-scores.json",
        out / "normal-diagnostics.json",
    ]
    if not all(p.is_file() for p in required):
        return False
    try:
        result = read_json(out / "result.json")
        return result.get("status") == "completed" and result.get("steps") == steps
    except Exception:
        return False


def main():
    queue = read_json(QUEUE)
    study = read_json(STUDY)

    selected = [
        t
        for t in queue["tasks"]
        if t["variant"] == "B2" and t["category"] in CATEGORIES and t["seed"] in SEEDS
    ]

    expected_keys = {(category, seed) for category in CATEGORIES for seed in SEEDS}
    actual_keys = {(t["category"], t["seed"]) for t in selected}

    if len(selected) != 24 or actual_keys != expected_keys:
        raise RuntimeError(f"Expected exactly 24 B2 tasks; found {len(selected)}")

    # Validate every task and every output before launching any process.
    for task in selected:
        category = task["category"]
        seed = task["seed"]
        if float(task["lambda_score"]) != LAMBDA:
            raise RuntimeError(f"Wrong lambda for {category}/seed-{seed}: {task['lambda_score']}")
        if task["steps"] != EXPECTED_STEPS[category]:
            raise RuntimeError(f"Wrong steps for {category}/seed-{seed}: {task['steps']}")

        manifest = ROOT / task["manifest"]
        if not manifest.is_file():
            raise RuntimeError(f"Missing manifest: {manifest}")
        if sha256(manifest) != task["manifest_sha256"]:
            raise RuntimeError(f"Manifest hash mismatch: {manifest}")

        out = OUT_ROOT / category / "B2" / f"seed-{seed}"
        if out.exists() and not complete(out, task["steps"]):
            raise RuntimeError(f"Incomplete output exists; refusing overwrite: {out}")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("B2 FULL-LENGTH EXPERIMENT", flush=True)
    print(f"Lambda: {LAMBDA}", flush=True)
    print(f"Categories: {CATEGORIES}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print("Training: 80 source passes; category-specific fixed steps", flush=True)
    print(f"Output root: {OUT_ROOT.relative_to(ROOT)}", flush=True)

    for category in CATEGORIES:
        tasks = sorted(
            [t for t in selected if t["category"] == category],
            key=lambda t: t["seed"],
        )

        jobs = []
        for task in tasks:
            seed = task["seed"]
            out = OUT_ROOT / category / "B2" / f"seed-{seed}"

            if complete(out, task["steps"]):
                print(f"Already complete; preserving {out}", flush=True)
                continue

            config = {
                "dataset": study["dataset"],
                "category": category,
                "manifest_sha256": task["manifest_sha256"],
                "variant": "B2",
                "lambda_score": LAMBDA,
                "seed": seed,
                "batch_size": study["batch_size"],
                "image_size": study["image_size"],
                "ad_resize": study["ad_resize"],
                "brightness": study["brightness"],
                "gamma": study["gamma"],
                "weights": "imagenet_v1",
            }

            config_path = CONFIG_DIR / (f"{category}-B2-lambda001-seed-{seed}.json")

            if config_path.exists():
                if read_json(config_path) != config:
                    raise RuntimeError(f"Config mismatch: {config_path}")
            else:
                tmp = config_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(config, indent=2) + "\n")
                tmp.replace(config_path)

            log_path = LOG_DIR / (f"b2-lambda001-{category}-seed-{seed}.log")
            if log_path.exists():
                raise RuntimeError(f"Log already exists; refusing overwrite: {log_path}")

            jobs.append((task, config_path, out, log_path))

        if not jobs:
            print(
                f"{category}: all three seeds already complete",
                flush=True,
            )
            continue

        print("=" * 60, flush=True)
        print(f"STARTING CATEGORY: {category}", flush=True)
        print(f"Seeds: {[j[0]['seed'] for j in jobs]}", flush=True)
        print(f"Steps per seed: {EXPECTED_STEPS[category]}", flush=True)

        processes = []
        started = time.monotonic()

        for task, config_path, out, log_path in jobs:
            command = [
                sys.executable,
                str(TRAINER),
                "--config",
                str(config_path),
                "--manifest",
                str(ROOT / task["manifest"]),
                "--root",
                str(Path.home() / "datasets/mvtec_ad_2"),
                "--output",
                str(out),
                "--steps",
                str(task["steps"]),
                "--device",
                "cuda",
                "--threads",
                "2",
                "--gpu-memory-fraction",
                "0.25",
                "--target-fpr",
                str(study["normal_validation_target_fpr"]),
                "--normal-diagnostics",
                "--cache-images",
            ]

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "2"
            env["PYTHONUNBUFFERED"] = "1"

            log = log_path.open("x")
            print(
                f"Launching {category}/seed-{task['seed']}: {out}",
                flush=True,
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                log.close()
                raise

            processes.append((task, process, log, log_path))

        failures = []
        for task, process, log, log_path in processes:
            code = process.wait()
            log.close()
            print(
                f"{category}/seed-{task['seed']} exited {code}; log: {log_path}",
                flush=True,
            )
            if code != 0:
                failures.append((task["seed"], code))

        results = []
        for task, _, _, _ in processes:
            out = OUT_ROOT / category / "B2" / f"seed-{task['seed']}"
            ok = complete(out, task["steps"])
            results.append(
                {
                    "seed": task["seed"],
                    "complete": ok,
                    "steps": task["steps"],
                    "output": str(out.relative_to(ROOT)),
                }
            )
            if not ok and not any(x[0] == task["seed"] for x in failures):
                failures.append((task["seed"], "missing/incomplete outputs"))

        summary = {
            "category": category,
            "variant": "B2",
            "lambda_score": LAMBDA,
            "seeds": SEEDS,
            "concurrent_processes": len(processes),
            "elapsed_seconds": time.monotonic() - started,
            "results": results,
            "failures": failures,
        }

        summary_path = LOG_DIR / f"b2-lambda001-{category}-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)

        if failures:
            raise SystemExit(f"{category}: at least one seed failed; stopping progression")

        print(
            f"{category}: ALL THREE SEEDS VERIFIED COMPLETE",
            flush=True,
        )

    print("=" * 60, flush=True)
    print("B2 LAMBDA 0.01 FULL STUDY FINISHED", flush=True)


if __name__ == "__main__":
    main()
