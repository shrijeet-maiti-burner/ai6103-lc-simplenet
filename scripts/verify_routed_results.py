"""Verify published aggregate counts, means and gates; no model inference."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(float(actual)), message)
    require(math.isclose(float(actual), float(expected), abs_tol=1e-12), message)


def read_rows(path, kind):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = set()
    cohorts = {}
    for row in rows:
        cohort = (row["category"], row["condition"]) if kind == "ad2" else (row["domain"],)
        key = (*cohort, row["method"], row["seed"])
        require(key not in keys, "duplicate aggregate row")
        keys.add(key)
        n0 = int(row["normal_images" if kind == "ad2" else "normal_count"])
        n1 = int(row["defective_images" if kind == "ad2" else "defect_count"])
        fp, tp = int(row["false_positives"]), int(row["true_positives"])
        require(n0 > 0 and n1 > 0 and 0 <= fp <= n0 and 0 <= tp <= n1, "invalid counts")
        require(cohorts.setdefault(cohort, (n0, n1)) == (n0, n1), "changed cohort")
        close(row["fpr"], fp / n0, "false-positive rate differs from counts")
        close(row["recall"], tp / n1, "recall differs from counts")
        auc = float(row["auroc" if kind == "ad2" else "image_auroc"])
        require(math.isfinite(auc) and 0 <= auc <= 1, "invalid auroc")
    return rows, keys


def gates(candidate, baseline):
    changed = "nonregular" if "nonregular" in candidate else "illumination"
    regular = "regular" if "regular" in candidate else "same"
    return {
        "lighting_fpr": baseline[changed]["fpr"] - candidate[changed]["fpr"] >= 0.05 - 1e-12,
        "lighting_recall": candidate[changed]["recall"] - baseline[changed]["recall"]
        >= -0.05 - 1e-12,
        "regular_fpr": candidate[regular]["fpr"] - baseline[regular]["fpr"] <= 0.02 + 1e-12,
        "regular_recall": candidate[regular]["recall"] - baseline[regular]["recall"]
        >= -0.05 - 1e-12,
    }


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    require(
        set(manifest["files"])
        == {
            "ad2-category-seed-metrics.csv",
            "blade-domain-seed-metrics.csv",
            "read-me-first.md",
            "result-summary.json",
            "routed-results.md",
            "verify_results.py",
        },
        "incomplete manifest",
    )
    for name, descriptor in manifest["files"].items():
        require(Path(name).name == name, "unsafe manifest member")
        raw = (folder / name).read_bytes()
        require(len(raw) == descriptor["bytes"], "file size differs: " + name)
        require(hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "hash differs: " + name)
    result = json.loads((folder / "result-summary.json").read_text(encoding="utf-8"))
    ad2, ad2_keys = read_rows(folder / "ad2-category-seed-metrics.csv", "ad2")
    blade, blade_keys = read_rows(folder / "blade-domain-seed-metrics.csv", "blade")
    categories = [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    ]
    ad2_methods = ["B0", "B1", "B2", "L0", "L1", "L2", "R0", "previous_B0"]
    blade_methods = ["B0", "B1", "B2", "L0", "L1", "L2", "L3", "R0", "R1", "R2", "P0", "P1"]
    domains = ["illumination", "same", "background", "view"]
    require(
        ad2_keys
        == {
            (c, g, a, str(s))
            for c in categories
            for g in ("regular", "nonregular")
            for a in ad2_methods
            for s in range(3)
        },
        "incomplete ad2 matrix",
    )
    require(
        blade_keys == {(g, a, str(s)) for g in domains for a in blade_methods for s in range(3)},
        "incomplete blade matrix",
    )
    ad2_means = {}
    for scope, cats in (
        ("all8", categories),
        ("without_sheet_metal", [c for c in categories if c != "sheet_metal"]),
    ):
        ad2_means[scope] = {}
        for arm in ad2_methods:
            ad2_means[scope][arm] = {}
            for group in ("regular", "nonregular"):
                selected = [
                    r
                    for r in ad2
                    if r["category"] in cats and r["method"] == arm and r["condition"] == group
                ]
                values = {
                    m: mean(float(r[m]) for r in selected) for m in ("fpr", "recall", "auroc")
                }
                ad2_means[scope][arm][group] = values
                for metric, value in values.items():
                    close(
                        value, result["ad2"]["means"][scope][arm][group][metric], "ad2 mean differs"
                    )
    blade_means = {}
    for arm in blade_methods:
        blade_means[arm] = {}
        for seed in range(3):
            thresholds = {
                r["threshold"] for r in blade if r["method"] == arm and r["seed"] == str(seed)
            }
            require(
                len(thresholds) == 1 and all(math.isfinite(float(q)) for q in thresholds),
                "blade threshold changed between domains",
            )
        for domain in domains:
            selected = [r for r in blade if r["method"] == arm and r["domain"] == domain]
            values = {
                m: mean(float(r[m]) for r in selected) for m in ("fpr", "recall", "image_auroc")
            }
            blade_means[arm][domain] = values
            for metric, value in values.items():
                close(value, result["blade"]["means"][domain][arm][metric], "blade mean differs")
    ad2_passes = []
    for baseline in ("B0", "previous_B0"):
        actual = gates(ad2_means["all8"]["R0"], ad2_means["all8"][baseline])
        require(
            actual == result["ad2"]["comparisons"]["all8"][baseline]["gates"], "ad2 gates differ"
        )
        ad2_passes.append(all(actual.values()))
    blade_gates = gates(blade_means["R0"], blade_means["B0"])
    require(
        all(ad2_passes) == result["ad2"]["original_target_point_qualified"], "ad2 decision differs"
    )
    require(blade_gates == result["blade"]["mean_gates"], "blade gates differ")
    require(
        all(blade_gates.values()) == result["blade"]["lighting_and_regular_mean_targets_met"],
        "blade decision differs",
    )
    ad2_decisions = sum(
        int(r["normal_images"]) + int(r["defective_images"])
        for r in ad2
        if r["method"] != "previous_B0"
    )
    blade_decisions = sum(int(r["normal_count"]) + int(r["defect_count"]) for r in blade)
    require(ad2_decisions == 22764 and blade_decisions == 59004, "decision count differs")
    return {
        "aggregate_verification_passed": True,
        "ad2_rows": len(ad2),
        "blade_rows": len(blade),
        "ad2_image_decisions": ad2_decisions,
        "blade_image_decisions": blade_decisions,
        "ad2_targets_met": all(ad2_passes),
        "blade_mean_targets_met": all(blade_gates.values()),
        "scope": (
            "published aggregate counts, means, fixed point gates and hashes; "
            "not model inference, raw-score replay or bootstrap recomputation"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(verify(parser.parse_args().folder), indent=2))


if __name__ == "__main__":
    main()
