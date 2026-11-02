import csv
import json

import pytest

from lc_simplenet.evaluate_runs import discover_runs, evaluate_all, run_matrix, write_outputs


def make_run(root, category, variant, seed, test=True, weight=None):
    path = root / category / variant / f"seed-{seed}"
    path.mkdir(parents=True)
    val = [{"image_id": f"v{i}", "label": 0, "score": float(i)} for i in range(5)]
    (path / "validation-scores.json").write_text(json.dumps(val))
    if weight is not None:
        (path / "run.json").write_text(json.dumps({"config": {"lambda_score": weight}}))
    if test:
        rows = [(1.0, 0), (4.0, 0), (5.0, 1), (2.0, 1)]
        data = [{"image_id": f"t{i}", "label": y, "score": s} for i, (s, y) in enumerate(rows)]
        (path / "test-scores.json").write_text(json.dumps(data))


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end_tables_and_missing_runs(tmp_path):
    root, out = tmp_path / "runs", tmp_path / "out"
    make_run(root, "can", "B0", 0)
    make_run(root, "can", "B0", 1)
    make_run(root, "can", "B1", 0, test=False)
    make_run(root, "can", "B2", 0, weight=0.01)
    runs = discover_runs(root)
    assert {r.method for r in runs} == {"B0", "B1", "B2@0.01"}
    matrix = run_matrix(runs, ["can"], [0, 1])
    status = {(r["method"], r["seed"]): r["status"] for r in matrix}
    assert status[("B0", 0)] == "ready" and status[("B1", 0)] == "validation_only"
    assert status[("B1", 1)] == "missing"
    cond = {"t0": "regular", "t1": "regular", "t2": "shifted", "t3": "shifted"}
    write_outputs(evaluate_all(runs, 0.25, cond), matrix, out, cond)

    thresholds = json.loads((out / "thresholds.json").read_text())
    assert thresholds["can/B1/seed-0"]["threshold"] == 3  # validation-only run still calibrated
    metrics = {(m["method"], m["seed"], m["condition"]): m for m in read(out / "metrics.csv")}
    assert ("B1", "0", "all") not in metrics  # no test scores, no test metric
    regular = metrics[("B0", "0", "regular")]
    assert (regular["false_positives"], regular["normal_images"]) == ("1", "2")
    assert regular["recall"] == ""  # no defective images: unavailable, not zero
    assert len(read(out / "predictions.csv")) == 3 * 4
    flagged = {p["image_id"]: p["flagged"] for p in read(out / "predictions.csv")[:4]}
    assert flagged == {"t0": "0", "t1": "1", "t2": "1", "t3": "0"}
    summary = [s for s in read(out / "summary.csv") if s["method"] == "B0"]
    assert {s["seeds"] for s in summary} == {"2"}


def test_flat_release_layout_is_read_from_run_json(tmp_path):
    path = tmp_path / "paired" / "can-B2-seed1"
    path.mkdir(parents=True)
    (path / "validation-scores.json").write_text(
        json.dumps([{"image_id": "v0", "label": 0, "score": 0.5}])
    )
    config = {"category": "can", "variant": "B2", "seed": 1, "lambda_score": 0.1}
    (path / "run.json").write_text(json.dumps({"config": config}))
    (run,) = discover_runs(tmp_path)
    assert (run.category, run.method, run.seed) == ("can", "B2@0.1", 1)


def test_thresholds_name_the_checkpoint_that_scored_the_test(tmp_path):
    root, out = tmp_path / "runs", tmp_path / "out"
    make_run(root, "can", "B0", 0)
    make_run(root, "can", "B1", 0, test=False)
    (root / "can/B0/seed-0/result.json").write_text(json.dumps({"checkpoint_sha256": "final"}))
    meta = {"checkpoint_file": "heads.pt", "checkpoint_sha256": "heads"}
    (root / "can/B0/seed-0/test-scores.meta.json").write_text(json.dumps(meta))
    runs = discover_runs(root)
    write_outputs(evaluate_all(runs, 0.25), run_matrix(runs, ["can"], [0]), out)
    thresholds = json.loads((out / "thresholds.json").read_text())
    assert thresholds["can/B0/seed-0"]["checkpoint_sha256"] == "final"
    assert thresholds["can/B0/seed-0"]["test_checkpoint"] == {"file": "heads.pt", "sha256": "heads"}
    assert thresholds["can/B1/seed-0"]["test_checkpoint"] is None


def test_duplicate_runs_rejected(tmp_path):
    make_run(tmp_path / "a", "can", "B0", 0, test=False)
    make_run(tmp_path / "b", "can", "B0", 0, test=False)
    with pytest.raises(ValueError):
        discover_runs(tmp_path)
