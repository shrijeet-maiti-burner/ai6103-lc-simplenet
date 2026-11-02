import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from lc_simplenet.pilot import config_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_evaluate_ad2():
    key = "ad2_scripts.evaluate_ad2"
    if key in sys.modules:
        return sys.modules[key]
    path = ROOT / "scripts" / "evaluate_ad2.py"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def stored_config(**overrides):
    data = {
        "dataset": "mvtec_ad2",
        "category": "fabric",
        "manifest_sha256": "a" * 64,
        "variant": "B0",
        "image_size": [256, 256],
        "seed": 0,
    }
    data.update(overrides)
    return data


def dummy_rows():
    return [
        {
            "image": "000.png",
            "label": 0,
            "image_score": 0.1,
            "threshold": 0.2,
            "predicted_anomaly": 0,
            "mask": "",
        }
    ]


def dummy_summary(evaluate_ad2, variant: str, cohort_id: str, **extra):
    return {
        **evaluate_ad2.cohort_metadata(cohort_id),
        "variant": variant,
        "checkpoint_config": stored_config(variant=variant),
        "checkpoint_config_sha256": "abc",
        "config_file": f"results/ad2_fabric_{variant.lower()}_config.json",
        "config_file_matches_checkpoint": False,
        "metrics": {
            "image_auroc": 0.5,
            "pixel_auroc_at_256": 0.4,
            "validation_threshold": 0.2,
            "image_fpr_at_threshold": 0.0,
            "image_recall_at_threshold": 0.0,
        },
        **extra,
    }


def write_run_tree(root: Path, *, omit: tuple[str, str] | None = None):
    for variant in ("B0", "B1", "B2"):
        lower = variant.lower()
        config = root / f"results/ad2_fabric_{lower}_config.json"
        run = root / f"results/ad2_fabric_{lower}_5000steps_recovery"
        config.parent.mkdir(parents=True, exist_ok=True)
        run.mkdir(parents=True, exist_ok=True)
        if omit != (variant, "config"):
            config.write_text("{}\n", encoding="utf-8")
        if omit != (variant, "checkpoint"):
            (run / "final.pt").write_bytes(b"ckpt")
        (run / "result.json").write_text(
            json.dumps({"calibration": {"threshold": 0.2}}) + "\n",
            encoding="utf-8",
        )


def leftover_partials(output: Path) -> list[Path]:
    return [
        path
        for path in output.parent.iterdir()
        if path.name.startswith(f".{output.name}.") and path.name.endswith(".partial")
    ]


def test_evaluation_paths_match_batch_output_directories():
    evaluate_ad2 = load_evaluate_ad2()
    batch = (ROOT / "scripts" / "run_ad2_fabric_batch.sh").read_text(encoding="utf-8")
    assert 'output="results/ad2_fabric_${lower}_5000steps_recovery"' in batch
    assert evaluate_ad2.DEFAULT_COHORT == "5000steps_recovery"
    assert not hasattr(evaluate_ad2, "RUNS")
    assert evaluate_ad2.COHORTS[evaluate_ad2.DEFAULT_COHORT]["runs"] == {
        "B0": ("results/ad2_fabric_b0_config.json", "results/ad2_fabric_b0_5000steps_recovery"),
        "B1": ("results/ad2_fabric_b1_config.json", "results/ad2_fabric_b1_5000steps_recovery"),
        "B2": ("results/ad2_fabric_b2_config.json", "results/ad2_fabric_b2_5000steps_recovery"),
    }
    assert evaluate_ad2.EXPLORATORY_100_STEP_RUNS == {
        "B0": ("results/ad2_fabric_b0_config.json", "results/ad2_fabric_b0_100steps"),
        "B1": ("results/ad2_fabric_b1_config.json", "results/ad2_fabric_b1_100steps"),
        "B2": ("results/ad2_fabric_b2_config.json", "results/ad2_fabric_b2_100steps"),
    }
    assert evaluate_ad2.COHORTS["100steps"]["output"] == "results/ad2_fabric_public_evaluation"
    assert (
        evaluate_ad2.COHORTS["5000steps_recovery"]["output"]
        == "results/ad2_fabric_5000steps_recovery_public_evaluation"
    )
    parser = evaluate_ad2.build_parser()
    assert parser.parse_args([]).cohort == "5000steps_recovery"
    assert parser.parse_args(["--cohort", "100steps"]).cohort == "100steps"


def test_evaluation_metadata_contains_public_test_disclosure():
    evaluate_ad2 = load_evaluate_ad2()
    metadata = evaluate_ad2.cohort_metadata("5000steps_recovery")
    disclosure = metadata["public_test_disclosure"]
    assert "test_public contains labelled public test data" in disclosure
    assert "100-step B0/B1/B2 exploratory checkpoints were evaluated" in disclosure
    assert "cannot establish lighting robustness" in disclosure
    assert "not AU-PRO" in disclosure
    assert "not official AD 2 benchmark results" in disclosure
    assert metadata["official_ad2_benchmark"] is False
    assert metadata["exploratory"] is True
    assert metadata["metric_note"] == "pixel_auroc_at_256 is pixel AUROC, not AU-PRO"
    assert "Aryaman" in metadata["formal_evaluation"]
    source = (ROOT / "scripts" / "evaluate_ad2.py").read_text(encoding="utf-8")
    assert "sys.path.insert" not in source
    assert "AU-PRO" in source
    assert "not official AD 2 benchmark results" in source
    assert source.lstrip().startswith('"""Local public-test scoring')


def test_missing_inputs_are_rejected_before_any_output(tmp_path):
    evaluate_ad2 = load_evaluate_ad2()
    write_run_tree(tmp_path, omit=("B2", "checkpoint"))
    output = tmp_path / "published"
    called = []

    def evaluate_one(resolved, *, cohort_id):
        called.append(resolved["variant"])
        return dummy_summary(evaluate_ad2, resolved["variant"], cohort_id), dummy_rows()

    with pytest.raises(FileNotFoundError, match="B2: missing checkpoint"):
        evaluate_ad2.evaluate_cohort(
            "5000steps_recovery",
            output=output,
            root=tmp_path,
            evaluate_one=evaluate_one,
        )
    write_run_tree(tmp_path)
    (tmp_path / "results/ad2_fabric_b1_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="B1: missing config"):
        evaluate_ad2.evaluate_cohort(
            "5000steps_recovery",
            output=output,
            root=tmp_path,
            evaluate_one=evaluate_one,
        )
    assert called == []
    assert not output.exists()
    assert leftover_partials(output) == []


def test_failed_variant_does_not_publish_partial_results(tmp_path):
    evaluate_ad2 = load_evaluate_ad2()
    write_run_tree(tmp_path)
    output = tmp_path / "published"
    called = []

    def evaluate_one(resolved, *, cohort_id):
        called.append(resolved["variant"])
        if resolved["variant"] == "B1":
            raise RuntimeError("B1 failed")
        return dummy_summary(evaluate_ad2, resolved["variant"], cohort_id), dummy_rows()

    with pytest.raises(RuntimeError, match="B1 failed"):
        evaluate_ad2.evaluate_cohort(
            "5000steps_recovery",
            output=output,
            root=tmp_path,
            evaluate_one=evaluate_one,
        )
    assert called == ["B0", "B1"]
    assert not output.exists()
    assert leftover_partials(output) == []


def test_successful_run_publishes_all_variants_and_comparison(tmp_path):
    evaluate_ad2 = load_evaluate_ad2()
    write_run_tree(tmp_path)
    output = tmp_path / "published"

    def evaluate_one(resolved, *, cohort_id):
        return dummy_summary(evaluate_ad2, resolved["variant"], cohort_id), dummy_rows()

    results = evaluate_ad2.evaluate_cohort(
        "5000steps_recovery",
        output=output,
        root=tmp_path,
        evaluate_one=evaluate_one,
    )
    assert [row["variant"] for row in results] == ["B0", "B1", "B2"]
    assert all(row["official_ad2_benchmark"] is False for row in results)
    assert all(row["exploratory"] is True for row in results)
    for variant in ("B0", "B1", "B2"):
        summary = json.loads((output / variant / "evaluation.json").read_text(encoding="utf-8"))
        assert summary["variant"] == variant
        assert (output / variant / "per_image.csv").is_file()
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert [row["variant"] for row in comparison] == ["B0", "B1", "B2"]
    assert all(row["official_ad2_benchmark"] is False for row in comparison)
    assert leftover_partials(output) == []


def test_completed_outputs_are_not_overwritten(tmp_path):
    evaluate_ad2 = load_evaluate_ad2()
    write_run_tree(tmp_path)
    output = tmp_path / "published"
    marker = {"variant": "B0", "keep": True}

    def evaluate_one(resolved, *, cohort_id):
        return dummy_summary(evaluate_ad2, resolved["variant"], cohort_id), dummy_rows()

    evaluate_ad2.evaluate_cohort(
        "5000steps_recovery",
        output=output,
        root=tmp_path,
        evaluate_one=evaluate_one,
    )
    (output / "B0" / "evaluation.json").write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_ad2.evaluate_cohort(
            "5000steps_recovery",
            output=output,
            root=tmp_path,
            evaluate_one=evaluate_one,
        )
    assert json.loads((output / "B0" / "evaluation.json").read_text(encoding="utf-8")) == marker


def test_checkpoint_provenance_records_stored_config(tmp_path):
    evaluate_ad2 = load_evaluate_ad2()
    saved = stored_config(seed=0, variant="B1")
    matching = tmp_path / "match.json"
    differing = tmp_path / "differ.json"
    matching.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    differing.write_text(json.dumps(stored_config(seed=7, variant="B1")) + "\n", encoding="utf-8")
    matched = evaluate_ad2.checkpoint_provenance(saved, variant="B1", config_file=matching)
    mismatched = evaluate_ad2.checkpoint_provenance(saved, variant="B1", config_file=differing)
    config = evaluate_ad2.coerce_pilot_config(saved)
    assert matched["checkpoint_config"]["seed"] == 0
    assert matched["checkpoint_config"]["variant"] == "B1"
    assert matched["checkpoint_config_sha256"] == config_hash(config)
    assert matched["checkpoint_config_sha256"] == evaluate_ad2.config_hash(config)
    assert matched["config_file"] == str(matching)
    assert matched["config_file_matches_checkpoint"] is True
    assert mismatched["config_file_matches_checkpoint"] is False
    assert mismatched["checkpoint_config"]["seed"] == 0
    assert mismatched["config_file"] == str(differing)
    with pytest.raises(ValueError, match="checkpoint variant is B1"):
        evaluate_ad2.checkpoint_provenance(saved, variant="B0", config_file=matching)
