#!/usr/bin/env python3
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
CONFIG_DIR = ROOT / "results/ad2-full-study/configs"
LOG_DIR = ROOT / "logs"

CATEGORY = os.environ.get("AD2_CATEGORY", "can")
VARIANT = os.environ.get("AD2_VARIANT", "B1")
SEEDS = [0, 1, 2]


def read_json(path):
    return json.loads(path.read_text())


def complete(task):
    out = ROOT / task["output"]
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
        return result.get("status") == "completed" and result.get("steps") == task["steps"]
    except Exception:
        return False


def main():
    queue = read_json(QUEUE)
    study = read_json(STUDY)

    selected = [
        t
        for t in queue["tasks"]
        if t["category"] == CATEGORY and t["variant"] == VARIANT and t["seed"] in SEEDS
    ]
    selected.sort(key=lambda t: t["seed"])

    if [t["seed"] for t in selected] != SEEDS:
        raise RuntimeError("Expected exactly seeds 0, 1, 2")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []

    # Validate every output and config before starting any process.
    for task in selected:
        out = ROOT / task["output"]
        if complete(task):
            print(f"Already complete; preserving: {out}")
            continue
        if out.exists():
            raise RuntimeError(f"Incomplete output exists; refusing to overwrite: {out}")

        config = {
            "dataset": study["dataset"],
            "category": task["category"],
            "manifest_sha256": task["manifest_sha256"],
            "variant": task["variant"],
            "lambda_score": task["lambda_score"],
            "seed": task["seed"],
            "batch_size": study["batch_size"],
            "image_size": study["image_size"],
            "ad_resize": study["ad_resize"],
            "brightness": study["brightness"],
            "gamma": study["gamma"],
            "weights": "imagenet_v1",
        }

        config_path = CONFIG_DIR / (f"{CATEGORY}-{VARIANT}-seed-{task['seed']}.json")

        if config_path.exists():
            if read_json(config_path) != config:
                raise RuntimeError(f"Config mismatch: {config_path}")
        else:
            tmp = config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(config, indent=2) + "\n")
            tmp.replace(config_path)

        jobs.append((task, config_path))

    if not jobs:
        print("All three seeds are already complete.")
        return

    print(f"Launching {len(jobs)} processes concurrently")
    print(f"Batch: {CATEGORY}/{VARIANT}")
    print(f"Seeds: {[t['seed'] for t, _ in jobs]}")
    print(f"Updates per seed: {jobs[0][0]['steps']}")

    processes = []
    started = time.monotonic()

    for task, config_path in jobs:
        seed = task["seed"]
        out = ROOT / task["output"]
        log_path = LOG_DIR / f"concurrent-{CATEGORY}-{VARIANT}-seed-{seed}.log"

        if log_path.exists():
            raise RuntimeError(f"Log already exists; refusing overwrite: {log_path}")

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

        log = log_path.open("x")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["PYTHONUNBUFFERED"] = "1"

        print(f"Starting seed {seed}: {out}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((task, process, log, log_path))

    failures = []
    for task, process, log, log_path in processes:
        code = process.wait()
        log.close()
        print(
            f"Seed {task['seed']} exited with code {code}; log: {log_path}",
            flush=True,
        )
        if code != 0:
            failures.append((task["seed"], code))

    elapsed = time.monotonic() - started

    results = []
    for task, _, _, _ in processes:
        out = ROOT / task["output"]
        ok = complete(task)
        results.append(
            {
                "seed": task["seed"],
                "complete": ok,
                "output": str(out.relative_to(ROOT)),
            }
        )
        if not ok and not any(x[0] == task["seed"] for x in failures):
            failures.append((task["seed"], "missing/incomplete outputs"))

    summary = {
        "category": CATEGORY,
        "variant": VARIANT,
        "seeds": SEEDS,
        "concurrent_processes": len(processes),
        "elapsed_seconds": elapsed,
        "results": results,
        "failures": failures,
    }

    summary_path = LOG_DIR / f"concurrent-{CATEGORY}-{VARIANT}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if failures:
        raise SystemExit("At least one seed failed; stopping batch progression.")

    print("ALL THREE SEEDS VERIFIED COMPLETE")


if __name__ == "__main__":
    main()
