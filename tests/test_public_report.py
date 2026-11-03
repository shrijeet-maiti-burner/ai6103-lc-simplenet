"""A report must not promote an incomplete or mismatched public evaluation."""

import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_public_report", Path(__file__).resolve().parents[1] / "scripts/build_public_report.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def inputs(tmp_path):
    frozen = {
        "categories": ["can", "sheet_metal"],
        "methods": ["B0", "B1"],
        "seeds": [0, 1, 2],
        "target_fpr": 0.05,
        "recall_tolerance": 0.05,
        "lighting_aggregates": True,
        "scope": "synthetic fixture; no measured performance",
        "gpu": "fixture",
        "scoring_git_commit": "fixture",
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(frozen))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    matrix = ["category,method,seed,status"] + [
        f"{c},{m},{s},ready"
        for c, m, s in itertools.product(frozen["categories"], frozen["methods"], frozen["seeds"])
    ]
    (evaluation / "run_matrix.csv").write_text("\n".join(matrix))
    rows = []
    for scope, categories in (
        ("all_categories", frozen["categories"]),
        ("without_sheet_metal", ["can"]),
    ):
        for method, condition in itertools.product(frozen["methods"], ["regular", "nonregular"]):
            rows.append(
                {
                    "scope": scope,
                    "categories": categories,
                    "expected_seeds": [0, 1, 2],
                    "method": method,
                    "condition": condition,
                    "status": "complete",
                    "fpr": {"mean": 0.1, "sd": 0.01, "n": 3},
                    "recall": {"mean": 0.8, "sd": 0.02, "n": 3},
                }
            )
    aggregate = {
        "status": "complete",
        "aggregation": "fixture",
        "recall_tolerance": 0.05,
        "macros": rows,
        "comparisons": [],
        "category_recall_checks": [],
    }
    (evaluation / "aggregate.json").write_text(json.dumps(aggregate))
    for name in ("thresholds.json", "metrics.csv", "predictions.csv"):
        (evaluation / name).write_text("fixture")
    return evaluation, protocol, tmp_path / "out", aggregate


def test_public_tables_are_separate_scopes_and_bound_to_inputs(inputs):
    evaluation, protocol, out, _ = inputs
    result = module.build(evaluation, protocol, out)
    assert len(result["input_sha256"]) == 6
    assert (out / "ad2-public-all_categories.tex").exists()
    assert (out / "ad2-public-without_sheet_metal.tex").exists()
    first_hash = result["input_sha256"]["predictions.csv"]
    (evaluation / "predictions.csv").write_text("changed")
    assert module.build(evaluation, protocol, out)["input_sha256"]["predictions.csv"] != first_hash


def test_a_missing_head_cannot_be_reported_as_complete(inputs):
    evaluation, protocol, out, _ = inputs
    matrix = evaluation / "run_matrix.csv"
    matrix.write_text(matrix.read_text().replace("can,B0,0,ready", "can,B0,0,missing"))
    with pytest.raises(ValueError, match="complete frozen study"):
        module.build(evaluation, protocol, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["missing_seed", "wrong_scope", "null", "nonfinite"])
def test_bad_estimates_never_produce_partial_report_tables(inputs, change):
    evaluation, protocol, out, aggregate = inputs
    modified = copy.deepcopy(aggregate)
    row = modified["macros"][0]
    if change == "missing_seed":
        row["fpr"]["n"] = 2
    elif change == "wrong_scope":
        row["categories"] = ["can"]
    else:
        row["fpr"]["mean"] = None if change == "null" else float("nan")
    (evaluation / "aggregate.json").write_text(json.dumps(modified))
    with pytest.raises(ValueError):
        module.build(evaluation, protocol, out)
    assert not out.exists()
