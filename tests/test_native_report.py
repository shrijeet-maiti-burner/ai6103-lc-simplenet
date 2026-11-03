import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "native_report", Path(__file__).resolve().parents[1] / "scripts/build_native_report.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cohort(tmp_path):
    native, scoring = tmp_path / "native", tmp_path / "scoring"
    frozen = {
        "categories": ["can", "sheet_metal"],
        "methods": ["B0", "B1"],
        "seeds": [0, 1, 2],
        "scoring_git_commit": "frozen-revision",
    }
    proto = dict(frozen, inputs={}, source_sha256={"engine": "retained-engine"})
    proto["frozen_image_protocol_sha256"] = write(scoring / "frozen-protocol.json", frozen)
    status = {"state": "complete", "expected": 12, "completed": [], "failed": [], "active": []}
    for cat, method, seed in itertools.product(
        frozen["categories"], frozen["methods"], frozen["seeds"]
    ):
        name = f"{cat}-{method}-seed{seed}"
        meta = {
            "category": cat,
            "variant": method,
            "seed": seed,
            "checkpoint_sha256": f"head-{name}",
            "run_config_sha256": f"config-{name}",
            "images": 2 if cat == "can" else 20,
            "image_ids_sha256": f"ids-{cat}",
            "scoring_git_commit": frozen["scoring_git_commit"],
        }
        inputs = {
            "test-scores.meta.json": write(scoring / "runs" / name / "test-scores.meta.json", meta),
            "test-scores.json": write(scoring / "runs" / name / "test-scores.json", []),
            "test-maps.npz": f"maps-{name}",
        }
        proto["inputs"][name] = inputs
        metric = {
            "images": meta["images"],
            "normal_pixels": 10 if cat == "can" else 100,
            "defect_pixels": 2,
            "regions": 1,
            "image_ids_sha256": meta["image_ids_sha256"],
            "mask_sha256": {f"{cat}/mask": "mask-hash"},
            "pixel_auroc": 0.2 + 0.1 * seed + (0.4 if cat == "sheet_metal" else 0),
            "au_pro": 0.1 + 0.05 * seed,
            "fpr_limit": 0.05,
            "connectivity": 8,
            "status": "complete",
            "test_scores_sha256": inputs["test-scores.json"],
            "test_maps_sha256": inputs["test-maps.npz"],
            "scoring_metadata_sha256": inputs["test-scores.meta.json"],
            **{
                k: meta[k] for k in ("checkpoint_sha256", "run_config_sha256", "scoring_git_commit")
            },
        }
        output_hash = write(native / "runs" / name / "native.json", metric)
        before = {
            "sources": {"engine": "retained-engine"},
            "fpr_limit": 0.05,
            "no_quantization": True,
            "test_map_sha256": inputs["test-maps.npz"],
            "test_score_sha256": inputs["test-scores.json"],
        }
        write(native / "runs" / name / "native.protocol.json", before)
        write(
            native / "runs" / name / "native.provenance.json",
            {
                "sources": before["sources"],
                "status": "complete",
                "metric_output_sha256": output_hash,
                "peak_rss_bytes": 1000,
            },
        )
        status["completed"].append({"identity": [cat, method, seed], "output_sha256": output_hash})
    write(native / "protocol.json", proto)
    write(native / "status.json", status)
    return native, scoring, tmp_path / "out"


def test_category_then_seed_aggregation_is_not_pixel_weighted(cohort):
    evidence = module.build(*cohort)
    all_cats, without = [r for r in evidence["macros"] if r["method"] == "B0"]
    assert all_cats["pixel_auroc"]["mean"] == pytest.approx(0.5)
    assert all_cats["pixel_auroc"]["sd"] == pytest.approx(0.1)
    assert all_cats["pixel_auroc"]["per_seed"] == pytest.approx([0.4, 0.5, 0.6])
    assert without["pixel_auroc"]["mean"] == pytest.approx(0.3)
    assert evidence["total_native_pixels_per_complete_sweep"] == 114


def test_missing_seed_is_rejected_instead_of_averaged_away(cohort):
    native, _, _ = cohort
    status = module.read(native / "status.json")
    status["completed"].pop()
    write(native / "status.json", status)
    with pytest.raises(ValueError, match="exactly once"):
        module.build(*cohort)


def test_stale_scoring_metadata_is_rejected(cohort):
    _, scoring, _ = cohort
    meta_path = scoring / "runs/can-B0-seed0/test-scores.meta.json"
    meta = module.read(meta_path)
    meta["checkpoint_sha256"] = "another-head"
    write(meta_path, meta)
    with pytest.raises(ValueError, match="bind"):
        module.build(*cohort)


def test_rehashed_population_change_still_fails_comparison(cohort):
    native, _, _ = cohort
    run = native / "runs/can-B0-seed0"
    metric = module.read(run / "native.json")
    metric["normal_pixels"] += 1
    output_hash = write(run / "native.json", metric)
    after = module.read(run / "native.provenance.json")
    after["metric_output_sha256"] = output_hash
    write(run / "native.provenance.json", after)
    status = module.read(native / "status.json")
    status["completed"][0]["output_sha256"] = output_hash
    write(native / "status.json", status)
    with pytest.raises(ValueError, match="populations differ"):
        module.build(*cohort)


def test_mutated_metric_bytes_fail_the_retained_hash(cohort):
    native, _, _ = cohort
    run = native / "runs/can-B0-seed0/native.json"
    metric = module.read(run)
    metric["au_pro"] = 0.99
    write(run, metric)
    with pytest.raises(ValueError, match="provenance changed"):
        module.build(*cohort)
