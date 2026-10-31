#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "results/ad2-full-study/ad2-task-queue.json"
RUNNER = ROOT / "scripts/ad2_concurrent_batch.py"

BATCHES = [
    ("fabric", "B0"),
    ("fabric", "B1"),
    ("fruit_jelly", "B0"),
    ("fruit_jelly", "B1"),
    ("rice", "B0"),
    ("rice", "B1"),
    ("sheet_metal", "B0"),
    ("sheet_metal", "B1"),
    ("vial", "B0"),
    ("vial", "B1"),
    ("wallplugs", "B0"),
    ("wallplugs", "B1"),
    ("walnuts", "B0"),
    ("walnuts", "B1"),
]

REQUIRED = [
    "final.pt",
    "run.json",
    "result.json",
    "validation-scores.json",
    "normal-diagnostics.json",
]


def complete(task):
    out = ROOT / task["output"]
    if not all((out / f).is_file() for f in REQUIRED):
        return False
    try:
        result = json.loads((out / "result.json").read_text())
        return result.get("status") == "completed" and result.get("steps") == task["steps"]
    except Exception:
        return False


def main():
    allocation = os.environ.get("AD2_ALLOCATION")
    if allocation not in ("1", "2"):
        raise SystemExit("Set AD2_ALLOCATION=1 or 2")

    selected_batches = BATCHES[:7] if allocation == "1" else BATCHES[7:]
    tasks = json.loads(QUEUE.read_text())["tasks"]

    print(f"ALLOCATION {allocation}: {len(selected_batches)} batches")

    # Preflight the complete allocation before launching training.
    for category, variant in selected_batches:
        batch = [
            t
            for t in tasks
            if t["category"] == category and t["variant"] == variant and t["seed"] in (0, 1, 2)
        ]
        if len(batch) != 3:
            raise RuntimeError(f"Expected 3 seeds: {category}/{variant}")

        for task in batch:
            out = ROOT / task["output"]
            if complete(task):
                print(f"VERIFIED DONE: {category}/{variant}/seed-{task['seed']}")
            elif out.exists():
                raise RuntimeError(f"Incomplete output; refusing overwrite: {out}")
            else:
                print(f"PENDING: {category}/{variant}/seed-{task['seed']}")

    # Protect the six previously completed can runs.
    for category, variant in [("can", "B0"), ("can", "B1")]:
        batch = [t for t in tasks if t["category"] == category and t["variant"] == variant]
        if len(batch) != 3 or not all(complete(t) for t in batch):
            raise RuntimeError(f"Previously completed {category}/{variant} failed verification")

    if "--dry-run" in sys.argv:
        print("DRY RUN PASSED. No training launched.")
        return

    for index, (category, variant) in enumerate(selected_batches, 1):
        print(f"\n===== {index}/{len(selected_batches)}: {category}/{variant} =====", flush=True)

        env = os.environ.copy()
        env["AD2_CATEGORY"] = category
        env["AD2_VARIANT"] = variant

        result = subprocess.run(
            [sys.executable, "-u", str(RUNNER)],
            cwd=ROOT,
            env=env,
        )

        if result.returncode != 0:
            raise SystemExit(f"STOPPED: {category}/{variant} failed")

        batch = [t for t in tasks if t["category"] == category and t["variant"] == variant]
        if not all(complete(t) for t in batch):
            raise SystemExit(f"STOPPED: verification failed for {category}/{variant}")

        print(f"VERIFIED: {category}/{variant}", flush=True)

    print(f"ALLOCATION {allocation} COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
