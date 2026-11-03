"""Evaluate finished runs at validation-calibrated thresholds and write results tables.

Example:
    python scripts/evaluate.py --runs results/ad2-full-study --out results/eval \
        --target-fpr 0.05 --categories can fabric --seeds 0 1 2

Runs without `test-scores.json` get a threshold and a `validation_only` row in the run matrix;
no test metric is produced for them. Optional `--conditions` is a JSON object mapping
`image_id` to its documented lighting condition.
"""

import argparse
import json
from pathlib import Path

from lc_simplenet.evaluate_runs import discover_runs, evaluate_all, run_matrix, write_outputs
from lc_simplenet.evaluation_summary import aggregate_results, filter_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, required=True, help="declared before testing")
    parser.add_argument("--categories", nargs="+", help="default: those found under --runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--methods", nargs="+", help="declared methods; missing runs stay visible")
    parser.add_argument("--recall-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--lighting-aggregates",
        action="store_true",
        help="also report nonregular/exposure/additional-light groups",
    )
    args = parser.parse_args()

    runs = discover_runs(args.runs)
    if not runs:
        raise SystemExit(f"No {'validation-scores.json'} found under {args.runs}")
    conditions = json.loads(args.conditions.read_text()) if args.conditions else None
    categories = args.categories or sorted({r.category for r in runs})
    runs = filter_runs(runs, categories, args.seeds, args.methods)
    matrix = run_matrix(runs, categories, args.seeds, args.methods or ())
    results = evaluate_all(runs, args.target_fpr, conditions, args.lighting_aggregates)
    aggregate = aggregate_results(results, matrix, args.recall_tolerance)
    write_outputs(results, matrix, args.out, conditions)
    (args.out / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    counts = {
        s: sum(r["status"] == s for r in matrix) for s in ("ready", "validation_only", "missing")
    }
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
