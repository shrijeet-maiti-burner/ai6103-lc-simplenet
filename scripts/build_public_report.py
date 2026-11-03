"""Render complete frozen public-subset image metrics; retain input hashes."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def build(evaluation: Path, protocol: Path, out: Path):
    frozen = json.loads(protocol.read_text(encoding="utf-8-sig"))
    aggregate = json.loads((evaluation / "aggregate.json").read_text(encoding="utf-8"))
    with (evaluation / "run_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    expected = set(itertools.product(frozen["categories"], frozen["methods"], frozen["seeds"]))
    identities = [(r["category"], r["method"], int(r["seed"])) for r in matrix]
    if (
        not expected
        or len(identities) != len(expected)
        or set(identities) != expected
        or any(r["status"] != "ready" for r in matrix)
        or aggregate["status"] != "complete"
        or frozen["target_fpr"] != 0.05
        or frozen["recall_tolerance"] != aggregate["recall_tolerance"]
        or not frozen["lighting_aggregates"]
    ):
        raise ValueError("public report requires the complete frozen study and operating point")
    thresholds = json.loads((evaluation / "thresholds.json").read_text(encoding="utf-8"))
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != {f"{c}/{m}/seed-{s}" for c, m, s in expected}
        or any(
            not isinstance(row, dict) or row.get("target_fpr") != frozen["target_fpr"]
            for row in thresholds.values()
        )
    ):
        raise ValueError("threshold evidence must cover every head at the frozen target FPR")
    rows = {(r["scope"], r["method"], r["condition"]): r for r in aggregate["macros"]}
    scopes = {"all_categories": frozen["categories"]}
    if "sheet_metal" in frozen["categories"]:
        scopes["without_sheet_metal"] = [c for c in frozen["categories"] if c != "sheet_metal"]
    tables = {}
    for scope, categories in scopes.items():
        lines = [
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Model & Light & FPR (\%) & Recall (\%) \\",
            r"\midrule",
        ]
        for method in frozen["methods"]:
            label = method if "@" not in method else r"B2 ($\lambda=" + method.split("@")[1] + "$)"
            for condition, light in [("regular", "R"), ("nonregular", "N")]:
                row = rows[scope, method, condition]
                if (
                    row["status"] != "complete"
                    or set(row["categories"]) != set(categories)
                    or set(row["expected_seeds"]) != set(frozen["seeds"])
                ):
                    raise ValueError("table row does not match the declared category/seed scope")
                values = []
                for metric in ("fpr", "recall"):
                    estimate = row[metric]
                    if estimate["n"] != len(frozen["seeds"]) or any(
                        v is None or not math.isfinite(v)
                        for v in (estimate["mean"], estimate["sd"])
                    ):
                        raise ValueError("missing or nonfinite public report estimate")
                    values.append(f"{100 * estimate['mean']:.1f}$\\pm${100 * estimate['sd']:.1f}")
                lines.append(f"{label} & {light} & " + " & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        tables[scope] = "\n".join(lines) + "\n"
    sources = {"protocol.json": protocol}
    sources.update(
        {
            name: evaluation / name
            for name in (
                "aggregate.json",
                "run_matrix.csv",
                "thresholds.json",
                "metrics.csv",
                "predictions.csv",
            )
        }
    )
    evidence = {
        "scope": frozen["scope"],
        "gpu": frozen["gpu"],
        "scoring_git_commit": frozen["scoring_git_commit"],
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        },
        "aggregation": aggregate["aggregation"],
        "recall_tolerance": aggregate["recall_tolerance"],
        "comparisons": aggregate["comparisons"],
        "category_recall_checks": aggregate["category_recall_checks"],
    }
    out.mkdir(parents=True, exist_ok=True)
    for scope, table in tables.items():
        (out / f"ad2-public-{scope}.tex").write_text(table, encoding="utf-8")
    (out / "public-evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/build/evidence"))
    args = parser.parse_args()
    build(args.evaluation, args.protocol, args.out)
    print("built complete public-subset image tables with input hashes")


if __name__ == "__main__":
    main()
