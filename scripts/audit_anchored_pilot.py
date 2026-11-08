"""Independently recount complete pilot predictions and fixed qualification gates.

No training/evaluation/gate functions are imported. This checks count and
decision integrity; it does not rerun inference or establish real-defect recall.
"""

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            h.update(block)
    return h.hexdigest()


def audit(root, role):
    protocol = load(root / "protocol.json")
    policy = protocol["policy"]
    confirmation = role == "confirmation"
    seeds = [policy["confirmation_seed"]] if confirmation else policy["seeds"]
    prefix = "confirmation" if confirmation else "screening"
    rows, hashes = [], {}
    for category, seed in itertools.product(policy["categories"], seeds):
        target = root / "training" / category / f"seed-{seed}"
        predictions_path = target / f"{role}-predictions.json"
        predictions = load(predictions_path)
        calibration = load(target / "calibration.json")
        if set(calibration) != set(policy["arms"]):
            raise AssertionError("missing calibration arm")
        for values in calibration.values():
            if len(values["scores"]) != policy["calibration_parents"]:
                raise AssertionError("incomplete calibration population")
            if not values["replay_bit_exact"]:
                raise AssertionError("checkpoint replay did not pass")
        parent_ids = [r["image_id"] for r in protocol["roles"][category][role]]
        expected = set(
            itertools.product(
                policy["arms"],
                parent_ids,
                range(len(policy["lights"])),
                ("normal", "spot", "scratch", "blend"),
            )
        )
        actual = {(r["arm"], r["parent_id"], r["light_index"], r["kind"]) for r in predictions}
        if len(predictions) != len(expected) or actual != expected:
            raise AssertionError("prediction population is incomplete or duplicated")
        for row in predictions:
            threshold = float(
                np.quantile(
                    calibration[row["arm"]]["scores"],
                    1 - policy["target_fpr"],
                    method="higher",
                )
            )
            if (row["category"], row["seed"], row["role"]) != (category, seed, role):
                raise AssertionError("prediction identity mismatch")
            if not math.isfinite(row["score"]) or threshold != row["threshold"]:
                raise AssertionError("nonfinite score or changed threshold")
            if row["label"] != int(row["kind"] != "normal") or row["flagged"] != (
                row["score"] > threshold
            ):
                raise AssertionError("label or frozen-threshold flag mismatch")
        for arm, condition in itertools.product(policy["arms"], ("identity", "corners")):
            group = [
                r
                for r in predictions
                if r["arm"] == arm
                and ((r["light_index"] == 0) if condition == "identity" else r["light_index"] > 0)
            ]
            normal = [r for r in group if r["kind"] == "normal"]
            changes = [r for r in group if r["kind"] != "normal"]
            auc = sum(
                1 if a["score"] > b["score"] else 0.5 if a["score"] == b["score"] else 0
                for a in changes
                for b in normal
            ) / (len(normal) * len(changes))
            rows.append(
                {
                    "category": category,
                    "seed": seed,
                    "arm": arm,
                    "condition": condition,
                    "normal": len(normal),
                    "synthetic_changes": len(changes),
                    "false_positives": sum(r["score"] > r["threshold"] for r in normal),
                    "true_positives": sum(r["score"] > r["threshold"] for r in changes),
                    "image_auroc": auc,
                    "erased_changes": sum(r["changed_pixels"] == 0 for r in changes),
                    "patch_std": calibration[arm]["patch_std"],
                    "gaussian_hinge": calibration[arm]["gaussian_hinge"],
                }
            )
        for path in [
            predictions_path,
            target / "calibration.json",
            *[target / f"{arm}.pt" for arm in policy["arms"]],
        ]:
            hashes[path.relative_to(root).as_posix()] = sha(path)

    def key(row):
        return row["category"], row["seed"], row["arm"], row["condition"]

    reported = sorted(load(root / f"{prefix}-metrics.json"), key=key)
    for actual, expected in zip(sorted(rows, key=key), reported, strict=True):
        if actual.keys() != expected.keys():
            raise AssertionError("metric columns changed")
        for name in actual:
            if name == "image_auroc":
                if abs(actual[name] - expected[name]) > 1e-12:
                    raise AssertionError("independent AUROC recount differs")
            elif actual[name] != expected[name]:
                raise AssertionError(f"independent metric recount differs: {name}")
    indexed = {key(r): r for r in rows}
    rule = policy["qualification"]
    per_seed, summary = [], []

    def rate(row, numerator, denominator):
        return Fraction(row[numerator], row[denominator])

    for seed in seeds:
        comparisons = {}
        for control in ("teacher", "identity", "augmentation", "anchor"):
            differences = {}
            worst = []
            for condition in ("identity", "corners"):
                pairs = [
                    (indexed[c, seed, "candidate", condition], indexed[c, seed, control, condition])
                    for c in policy["categories"]
                ]
                differences[condition] = {}
                for metric, numerator, denominator in (
                    ("fpr", "false_positives", "normal"),
                    ("recall", "true_positives", "synthetic_changes"),
                ):
                    deltas = [
                        rate(a, numerator, denominator) - rate(b, numerator, denominator)
                        for a, b in pairs
                    ]
                    differences[condition][metric] = sum(deltas, Fraction()) / len(pairs)
                    if condition == "corners" and metric == "recall":
                        worst = deltas
            clean, corner = differences["identity"], differences["corners"]
            required = (
                rule["corner_fpr_drop_vs_anchor"]
                if control == "anchor"
                else rule["corner_fpr_drop_vs_identity_and_augmentation"]
            )
            passed = all(
                (
                    corner["fpr"] <= -Fraction(str(required)),
                    corner["recall"] >= 0,
                    clean["fpr"] <= Fraction(str(rule["clean_fpr_increase_max"])),
                    clean["recall"] >= -Fraction(str(rule["clean_recall_drop_max"])),
                    min(worst) >= -Fraction(str(rule["worst_category_corner_recall_drop_max"])),
                )
            )
            comparisons[control] = {
                "deltas": {
                    c: {k: float(v) for k, v in values.items()} for c, values in differences.items()
                },
                "passed": passed,
            }
        good_categories = 0
        eligible = True
        for category in policy["categories"]:
            candidate = indexed[category, seed, "candidate", "corners"]
            augmentation = indexed[category, seed, "augmentation", "corners"]
            good_categories += rate(candidate, "false_positives", "normal") < rate(
                augmentation, "false_positives", "normal"
            )
            eligible &= (
                candidate["patch_std"]
                >= rule["minimum_calibration_patch_std_ratio_vs_augmentation"]
                * augmentation["patch_std"]
            )
            eligible &= (
                candidate["gaussian_hinge"]
                <= rule["maximum_calibration_gaussian_hinge_ratio_vs_augmentation"]
                * augmentation["gaussian_hinge"]
            )
        per_seed.append(
            {
                "seed": seed,
                "comparisons": comparisons,
                "eligible": bool(eligible),
                "categories_fpr_improved": good_categories,
                "passed": bool(
                    eligible
                    and good_categories
                    >= rule["categories_corner_fpr_improved_vs_augmentation_min"]
                    and all(r["passed"] for r in comparisons.values())
                ),
            }
        )
        for arm, condition in itertools.product(policy["arms"], ("identity", "corners")):
            group = [indexed[c, seed, arm, condition] for c in policy["categories"]]
            summary.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "condition": condition,
                    "macro_fpr": float(
                        sum((rate(r, "false_positives", "normal") for r in group), Fraction())
                        / len(group)
                    ),
                    "macro_synthetic_recall": float(
                        sum(
                            (rate(r, "true_positives", "synthetic_changes") for r in group),
                            Fraction(),
                        )
                        / len(group)
                    ),
                }
            )
    decision = load(root / f"{prefix}-decision.json")
    if decision["per_seed"] != per_seed or decision["passed"] != all(r["passed"] for r in per_seed):
        raise AssertionError("independent qualification decision differs")
    return {
        "scope": (
            "independent counts/thresholds/AUROC/gates; no inference replay or real-defect claim"
        ),
        "protocol_sha256": sha(root / "protocol.json"),
        "metrics": len(rows),
        "decision_reproduced": True,
        "passed": decision["passed"],
        "macro_rates": summary,
        "artifact_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.role)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print({"metrics": result["metrics"], "decision_reproduced": True, "passed": result["passed"]})


if __name__ == "__main__":
    main()
