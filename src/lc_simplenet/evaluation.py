"""Frozen-threshold counts, image AUROC and seed summaries.

Scores follow the repository convention: higher means more anomalous and an image is flagged
when `score > threshold`. Thresholds come from normal regular-light validation scores only and
are reused unchanged for every test condition. Missing denominators give None, never zero.
Native pixel AUROC and AU-PRO are implemented in localization.py.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from lc_simplenet.metrics import image_counts, normal_validation_threshold

METRIC_VERSION = "d-eval-1"
RATE_KEYS = ("fpr", "recall", "image_auroc")


def load_scores(path: str | Path) -> list[dict]:
    """Read a run's score rows (`image_id`, `label`, `score`); reject duplicate IDs."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of score rows")
    for row in rows:
        if not {"image_id", "label", "score"} <= row.keys():
            raise ValueError(f"{path}: row lacks image_id, label or score: {row}")
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate image_id")
    return rows


def image_auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), ties count one half. None if a class is absent."""
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels)
    if values.shape != truth.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite vector matching labels")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Labels must contain only 0 or 1")
    n_pos, n_neg = int((truth == 1).sum()), int((truth == 0).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:  # average ranks over tied scores
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    u = ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def calibrate(validation_rows: Sequence[Mapping], target_fpr: float) -> dict:
    """Threshold on normal validation scores; records the achieved rate and sample size."""
    labels = [row["label"] for row in validation_rows]
    if any(label != 0 for label in labels):
        raise ValueError("Calibration uses normal validation images only (label 0)")
    scores = [row["score"] for row in validation_rows]
    threshold = normal_validation_threshold(scores, target_fpr)
    achieved = image_counts(scores, labels, threshold)
    return {
        "method": "higher empirical quantile; strict score > threshold",
        "target_fpr": target_fpr,
        "threshold": threshold,
        "validation_images": achieved["normal_images"],
        "validation_false_positives": achieved["false_positives"],
        "validation_fpr": achieved["fpr"],
    }


def evaluate_condition(rows: Sequence[Mapping], threshold: float) -> dict:
    """Counts, FPR, recall and image AUROC for one set of labelled test rows."""
    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]
    result = image_counts(scores, labels, threshold)
    result["image_auroc"] = image_auroc(scores, labels)
    return result


def evaluate_run(
    validation_rows: Sequence[Mapping],
    test_rows: Sequence[Mapping],
    target_fpr: float,
    conditions: Mapping[str, str] | None = None,
    lighting_aggregates: bool = False,
) -> dict:
    """One model/category/seed: calibrate once, then score every condition at that threshold.

    `conditions` maps image_id to a documented lighting condition. When given it must cover
    every test row; without it only the pooled `all` group is reported (no invented groups).
    """
    calibration = calibrate(validation_rows, target_fpr)
    threshold = calibration["threshold"]
    groups = {"all": list(test_rows)}
    if conditions is not None:
        missing = [row["image_id"] for row in test_rows if row["image_id"] not in conditions]
        if missing:
            raise ValueError(f"{len(missing)} test images lack a documented condition")
        for row in test_rows:
            if conditions[row["image_id"]] in {
                "all",
                "nonregular",
                "exposure_changed",
                "additional_light",
            }:
                raise ValueError("condition map uses a reserved aggregate group name")
            groups.setdefault(conditions[row["image_id"]], []).append(row)
        if lighting_aggregates:
            rules = {
                "nonregular": lambda name: name != "regular",
                "exposure_changed": lambda name: name in {"underexposed", "overexposed"},
                "additional_light": lambda name: name.startswith("shift_"),
            }
            for name, include in rules.items():
                selected = [row for row in test_rows if include(conditions[row["image_id"]])]
                if selected:
                    groups[name] = selected
    return {
        "metric_version": METRIC_VERSION,
        "calibration": calibration,
        "conditions": {name: evaluate_condition(rows, threshold) for name, rows in groups.items()},
    }


def mean_sd(values: Sequence[float | None]) -> dict:
    """Mean and sample SD (ddof=1) over seeds. Any missing seed makes the summary unavailable."""
    if not values or any(v is None for v in values):
        return {"mean": None, "sd": None, "n": len(values)}
    array = np.asarray(values, dtype=np.float64)
    sd = float(array.std(ddof=1)) if array.size > 1 else None
    return {"mean": float(array.mean()), "sd": sd, "n": int(array.size)}


def summarize_seeds(runs: Mapping[int, dict]) -> dict:
    """Per-condition mean/SD of FPR, recall and AUROC over seeds, with per-seed counts kept."""
    if not runs:
        raise ValueError("No seed results to summarize")
    names = {frozenset(run["conditions"]) for run in runs.values()}
    if len(names) != 1:
        raise ValueError("Seeds report different condition sets")
    summary = {}
    for condition in next(iter(runs.values()))["conditions"]:
        per_seed = {seed: run["conditions"][condition] for seed, run in sorted(runs.items())}
        summary[condition] = {
            "seeds": sorted(per_seed),
            "per_seed": per_seed,
            **{key: mean_sd([r[key] for r in per_seed.values()]) for key in RATE_KEYS},
        }
    return summary


def macro_average(
    category_summaries: Mapping[str, dict], condition: str, key: str, exclude: Sequence[str] = ()
) -> dict:
    """Unweighted mean over categories of each category's seed-mean (not pooled images)."""
    means = [
        summary[condition][key]["mean"]
        for category, summary in sorted(category_summaries.items())
        if category not in exclude
    ]
    if not means or any(m is None for m in means):
        return {"mean": None, "categories": len(means)}
    return {"mean": float(np.mean(means)), "categories": len(means)}
