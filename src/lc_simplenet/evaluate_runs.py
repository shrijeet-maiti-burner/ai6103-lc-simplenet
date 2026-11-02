"""Discover finished runs, freeze thresholds and write the evaluation tables.

Run layout: `<root>/<category>/<variant>/seed-<n>/` holding `validation-scores.json` (written
by training) and, once test scoring exists, `test-scores.json` with the same row format. B2
runs are labelled `B2@<weight>` from `run.json`. Nothing here selects a checkpoint, weight or
threshold from test data: thresholds come from normal validation only.
"""

import csv
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from lc_simplenet.evaluation import (
    METRIC_VERSION,
    RATE_KEYS,
    calibrate,
    evaluate_run,
    load_scores,
    summarize_seeds,
)

VALIDATION_FILE = "validation-scores.json"
TEST_FILE = "test-scores.json"
COUNT_KEYS = ("false_positives", "normal_images", "true_positives", "defective_images")


@dataclass(frozen=True)
class RunRef:
    category: str
    method: str
    seed: int
    path: Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(variant: str, weight) -> str:
    return f"B2@{weight:g}" if variant == "B2" and weight is not None else variant


def _identify(path: Path) -> tuple[str, str, int] | None:
    """(category, method, seed) from `run.json`, else from `<category>/<variant>/seed-<n>`.

    `run.json` is preferred so any directory layout works, e.g. the release's flat
    `<category>-<variant>-seed<k>` folders.
    """
    run_file = path / "run.json"
    if run_file.exists():
        config = json.loads(run_file.read_text(encoding="utf-8")).get("config", {})
        if {"category", "variant", "seed"} <= config.keys():
            return (
                config["category"],
                _label(config["variant"], config.get("lambda_score")),
                int(config["seed"]),
            )
    match = re.fullmatch(r"seed-(\d+)", path.name)
    if not match or path.parent.parent == path.parent:
        return None
    weight = None
    if run_file.exists():
        weight = (
            json.loads(run_file.read_text(encoding="utf-8")).get("config", {}).get("lambda_score")
        )
    return path.parent.parent.name, _label(path.parent.name, weight), int(match[1])


def discover_runs(root: Path) -> list[RunRef]:
    runs = []
    for scores in sorted(root.rglob(VALIDATION_FILE)):
        identity = _identify(scores.parent)
        if identity:
            runs.append(RunRef(*identity, scores.parent))
    keys = [(r.category, r.method, r.seed) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("Two run directories share a category, method and seed")
    return runs


def run_matrix(runs: list[RunRef], categories, seeds, methods=()) -> list[dict]:
    """Expected category x method x seed grid with what exists. Missing runs stay visible."""
    found = {(r.category, r.method, r.seed): r for r in runs}
    names = sorted({r.method for r in runs} | {"B0", "B1"} | set(methods))
    rows = []
    for category, method, seed in itertools.product(categories, names, seeds):
        run = found.get((category, method, seed))
        if run is None:
            status = "missing"
        else:
            status = "ready" if (run.path / TEST_FILE).exists() else "validation_only"
        rows.append({"category": category, "method": method, "seed": seed, "status": status})
    return rows


def evaluate_all(runs: list[RunRef], target_fpr: float, conditions=None) -> dict:
    """Evaluate each run once; runs without test scores contribute thresholds only."""
    results = {}
    for run in runs:
        validation = load_scores(run.path / VALIDATION_FILE)
        test_file = run.path / TEST_FILE
        test = load_scores(test_file) if test_file.exists() else []
        evaluated = evaluate_run(validation, test, target_fpr, conditions) if test else None
        calibration = calibrate(validation, target_fpr)
        result_file = run.path / "result.json"
        checkpoint = (
            json.loads(result_file.read_text(encoding="utf-8")).get("checkpoint_sha256")
            if result_file.exists()
            else None
        )
        results[(run.category, run.method, run.seed)] = {
            "calibration": calibration,
            "evaluation": evaluated,
            "validation_scores_sha256": sha256(run.path / VALIDATION_FILE),
            "test_scores_sha256": sha256(test_file) if test else None,
            "checkpoint_sha256": checkpoint,
            "test": test,
        }
    return results


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict, matrix: list[dict], out: Path, conditions=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "run_matrix.csv", ["category", "method", "seed", "status"], matrix)

    thresholds = {
        f"{c}/{m}/seed-{s}": {
            **r["calibration"],
            "validation_scores_sha256": r["validation_scores_sha256"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "metric_version": METRIC_VERSION,
        }
        for (c, m, s), r in sorted(results.items())
    }
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    predictions, metrics = [], []
    for (category, method, seed), r in sorted(results.items()):
        if not r["evaluation"]:
            continue
        threshold = r["calibration"]["threshold"]
        for row in r["test"]:
            predictions.append(
                {
                    "category": category,
                    "method": method,
                    "seed": seed,
                    "image_id": row["image_id"],
                    "label": row["label"],
                    "score": row["score"],
                    "threshold": threshold,
                    "flagged": int(row["score"] > threshold),
                    "condition": (conditions or {}).get(row["image_id"], ""),
                }
            )
        for condition, values in r["evaluation"]["conditions"].items():
            metrics.append(
                {
                    "category": category,
                    "method": method,
                    "seed": seed,
                    "condition": condition,
                    "threshold": threshold,
                    **{k: values[k] for k in COUNT_KEYS},
                    **{k: values[k] for k in RATE_KEYS},
                    "test_scores_sha256": r["test_scores_sha256"],
                    "metric_version": METRIC_VERSION,
                }
            )
    write_csv(
        out / "predictions.csv",
        ["category", "method", "seed", "image_id", "label", "score", "threshold", "flagged"]
        + ["condition"],
        predictions,
    )
    metric_fields = (
        ["category", "method", "seed", "condition", "threshold"]
        + list(COUNT_KEYS)
        + list(RATE_KEYS)
        + ["test_scores_sha256", "metric_version"]
    )
    write_csv(out / "metrics.csv", metric_fields, metrics)

    summary = []
    groups = {}
    for (category, method, seed), r in results.items():
        if r["evaluation"]:
            groups.setdefault((category, method), {})[seed] = r["evaluation"]
    for (category, method), seeds in sorted(groups.items()):
        for condition, values in summarize_seeds(seeds).items():
            summary.append(
                {
                    "category": category,
                    "method": method,
                    "condition": condition,
                    "seeds": len(values["seeds"]),
                    **{
                        f"{k}_{stat}": values[k][stat] for k in RATE_KEYS for stat in ("mean", "sd")
                    },
                }
            )
    summary_fields = ["category", "method", "condition", "seeds"] + [
        f"{k}_{stat}" for k in RATE_KEYS for stat in ("mean", "sd")
    ]
    write_csv(out / "summary.csv", summary_fields, summary)
