"""CPU checks for the can pilot gate. No head, dataset, GPU, or selector."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lc_simplenet.score_field import RELIGHT_NAMES


def load_pilot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_can_score_field_pilot.py"
    spec = importlib.util.spec_from_file_location("run_can_score_field_pilot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_pilot()


def config_dict(**overrides) -> dict:
    raw = {
        "dataset": "mvtec_ad2",
        "category": "can",
        "manifest_sha256": "b" * 64,
        "variant": "B0",
        "lambda_score": 0,
        "seed": 0,
        "batch_size": 8,
        "image_size": [256, 256],
        "ad_resize": None,
        "noise_std": 0.015,
        "margin": 0.5,
        "adapter_lr": 0.0001,
        "discriminator_lr": 0.0002,
        "brightness": [0.8, 1.2],
        "gamma": [0.8, 1.2],
        "weights": "imagenet_v1",
    }
    raw.update(overrides)
    return raw


def score_rows(count: int = 46, stem: str = "regular") -> list[dict]:
    return [
        {
            "image_id": f"can/validation/good/{index:03d}_{stem}.png",
            "label": 0,
            "score": 0.25,
        }
        for index in range(count)
    ]


def six_rows() -> dict[str, list[dict]]:
    return {name: score_rows() for name in pilot.CAN_RUN_NAMES}


def make_runs(predict_value: float = 0.25, fail_name: str | None = None):
    calls = []

    def build(name: str):
        variant, seed = pilot.parse_can_run(name)
        saved = 1.0 if name == fail_name else predict_value

        def predict(images):
            calls.append(name)
            return np.full((1, 4, 4), predict_value, dtype=np.float32)

        image = np.full((8, 8, 3), 0.4, dtype=np.float32)
        row = {"image_id": "can/validation/good/000_regular.png", "label": 0, "score": saved}
        return pilot.PreparedRun(name, variant, seed, [row], [image], predict)

    return [build(name) for name in pilot.CAN_RUN_NAMES], calls


def test_write_json_renders_nested_paths_as_strings(tmp_path):
    destination = tmp_path / "environment.json"
    staged = tmp_path / "torch-hub" / "checkpoints" / pilot.WEIGHT_FILENAME
    maps = tmp_path / "maps" / "can-B0-seed0.npz"
    pilot.write_json(
        destination,
        {
            "safeguards": {
                "backbone_sha256": "abc123",
                "staged": staged,
                "by_run": {tmp_path / "can-B0-seed0": [maps]},
            },
            "status": "completed",
        },
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["safeguards"]["backbone_sha256"] == "abc123"
    assert loaded["safeguards"]["staged"] == os.fspath(staged)
    assert loaded["safeguards"]["by_run"][os.fspath(tmp_path / "can-B0-seed0")] == [os.fspath(maps)]
    assert loaded["status"] == "completed"


def test_python_and_uv_must_match_the_can_study():
    pilot.assert_locked_runtime(pilot.locked_runtime_record())
    other_python = pilot.locked_runtime_record()
    other_python["python"] = "3.12.14"
    with pytest.raises(pilot.PilotFailure, match="python"):
        pilot.assert_locked_runtime(other_python)
    other_uv = pilot.locked_runtime_record()
    other_uv["uv"] = "0.13.5"
    with pytest.raises(pilot.PilotFailure, match="uv"):
        pilot.assert_locked_runtime(other_uv)
    assert pilot.parse_uv_version("uv 0.13.0 (4ecd0eff2 2026-10-09)") == "0.13.0"


def test_provenance_is_verified_before_any_scored_file_changes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sample = source / "src" / "example.py"
    sample.parent.mkdir()
    sample.write_text("print('can pilot')\n", encoding="utf-8")
    digest = pilot.sha256_file(sample)
    manifest = {
        "git_head": "1f4de6afc1911a3ad44c03a33cbaeb6296aff66c",
        "submodule": {"commit": "351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b"},
        "files": {"src/example.py": digest},
    }
    manifest_path = source / "provenance" / "can-pilot-code.json"
    manifest_path.parent.mkdir()
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(payload)
    sidecar = manifest_path.with_suffix(".sha256")
    sidecar.write_text(
        f"{pilot.hashlib.sha256(payload).hexdigest()}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    verified = pilot.verify_provenance(source, manifest_path, sidecar)
    assert verified["git_head"] == manifest["git_head"]
    assert verified["sha256"] == pilot.sha256_file(manifest_path)
    sample.write_text("changed\n", encoding="utf-8")
    with pytest.raises(pilot.PilotFailure, match="hash mismatch"):
        pilot.verify_provenance(source, manifest_path, sidecar)
    sidecar.write_text(f"{'0' * 64}  {manifest_path.name}\n", encoding="utf-8")
    with pytest.raises(pilot.PilotFailure, match="sidecar"):
        pilot.verify_provenance(source, manifest_path, sidecar)


def test_locked_runtime_accepts_only_the_recorded_stack():
    pilot.assert_locked_runtime(pilot.locked_runtime_record())
    mismatched = pilot.locked_runtime_record()
    mismatched["torch"] = "2.11.0+cu128"
    mismatched["gpu"] = "NVIDIA L40S"
    with pytest.raises(pilot.PilotFailure, match="torch"):
        pilot.assert_locked_runtime(mismatched)
    offline = pilot.locked_runtime_record()
    offline["cuda_available"] = False
    offline["gpu_count"] = 0
    offline["gpu"] = ""
    with pytest.raises(pilot.PilotFailure, match="cuda_unavailable"):
        pilot.assert_locked_runtime(offline)


def test_numpy_and_pillow_pins_are_part_of_the_runtime_gate():
    mismatched = pilot.locked_runtime_record()
    mismatched["numpy"] = "2.0.0"
    mismatched["pillow"] = "10.0.0"
    with pytest.raises(pilot.PilotFailure, match="numpy"):
        pilot.assert_locked_runtime(mismatched)


def test_backbone_checksum_rejects_a_different_file_before_the_hub_changes(tmp_path, monkeypatch):
    weight = tmp_path / pilot.WEIGHT_FILENAME
    weight.write_bytes(b"not-the-imagenet-backbone")
    before = torch.hub.get_dir()
    with pytest.raises(pilot.PilotFailure, match="checksum"):
        pilot.stage_verified_weights(weight, tmp_path / "hub")
    assert torch.hub.get_dir() == before

    digest = pilot.sha256_file(weight)
    monkeypatch.setattr(pilot, "WEIGHT_SHA256", digest)
    try:
        staged = pilot.stage_verified_weights(weight, tmp_path / "hub")
        assert staged.name == pilot.WEIGHT_FILENAME
        assert pilot.sha256_file(staged) == digest
        assert weight.read_bytes() == b"not-the-imagenet-backbone"
    finally:
        torch.hub.set_dir(before)


def test_encoder_and_head_checksums_reject_a_different_file(tmp_path, monkeypatch):
    pilot.assert_encoder_hash(pilot.ENCODER_SHA256)
    with pytest.raises(pilot.PilotFailure, match="encoder"):
        pilot.assert_encoder_hash("0" * 64)

    head = tmp_path / "heads.pt"
    head.write_bytes(b"different-head")
    with pytest.raises(pilot.PilotFailure, match="checksum"):
        pilot.assert_head_file(head, "can-B0-seed0")
    monkeypatch.setitem(pilot.HEAD_SHA256, "can-B0-seed0", pilot.sha256_file(head))
    assert pilot.assert_head_file(head, "can-B0-seed0") == pilot.sha256_file(head)


def test_released_config_rejects_b2_and_a_run_json_mismatch():
    accepted = config_dict()
    pilot.assert_released_head_config(accepted, accepted, "can-B0-seed0")
    with pytest.raises(pilot.PilotFailure, match="B0 and B1"):
        pilot.assert_released_head_config(
            config_dict(variant="B2", lambda_score=1.0),
            config_dict(variant="B2", lambda_score=1.0),
            "can-B0-seed0",
        )
    with pytest.raises(pilot.PilotFailure, match="differs"):
        pilot.assert_released_head_config(config_dict(seed=1), config_dict(seed=0), "can-B0-seed0")


def test_score_rows_require_46_shared_validation_ids():
    pilot.assert_can_score_rows(six_rows())
    short = six_rows()
    short["can-B1-seed2"] = score_rows(45)
    with pytest.raises(pilot.PilotFailure, match="45"):
        pilot.assert_can_score_rows(short)
    leaked = six_rows()
    leaked["can-B0-seed0"][0]["image_id"] = "can/test_public/good/000_regular.png"
    with pytest.raises(pilot.PilotFailure, match="test_public"):
        pilot.assert_can_score_rows(leaked)
    drifted = six_rows()
    drifted["can-B1-seed1"] = score_rows(stem="other")
    with pytest.raises(pilot.PilotFailure, match="same validation image ids"):
        pilot.assert_can_score_rows(drifted)


def test_heads_root_must_be_exactly_the_six_can_runs(tmp_path):
    root = tmp_path / "paired"
    root.mkdir()
    for name in pilot.CAN_RUN_NAMES:
        (root / name).mkdir()
    assert set(pilot.assert_heads_layout(root)) == set(pilot.CAN_RUN_NAMES)
    (root / "can-B2-seed0").mkdir()
    with pytest.raises(pilot.PilotFailure, match="can-B2-seed0"):
        pilot.assert_heads_layout(root)


def test_a_failed_regular_check_writes_no_relight_maps(tmp_path, monkeypatch):
    runs, calls = make_runs(fail_name="can-B1-seed2")
    relight_calls = []

    def refuse_relight(_image):
        relight_calls.append(1)
        raise AssertionError("relight must not run after a failed regular-light check")

    monkeypatch.setattr(pilot, "relight_validation_image", refuse_relight)
    report = {}
    with pytest.raises(pilot.PilotFailure, match="1e-3"):
        pilot.execute(runs, tmp_path, report)
    checks = json.loads((tmp_path / "regular-checks.json").read_text(encoding="utf-8"))
    assert checks["passed"] is False
    assert [row["name"] for row in checks["runs"]] == list(pilot.CAN_RUN_NAMES)
    assert sum(not row["passed"] for row in checks["runs"]) == 1
    assert checks["runs"][-1]["name"] == "can-B1-seed2"
    assert report["maps_written"] == 0
    assert relight_calls == []
    assert calls == list(pilot.CAN_RUN_NAMES)
    assert not (tmp_path / "maps").exists()
    assert list(tmp_path.glob("*.npz")) == []
    assert list(tmp_path.glob("**/*.npz")) == []


def test_all_six_passes_write_six_maps_after_regular_checks(tmp_path, monkeypatch):
    runs, calls = make_runs()
    relight_calls = []
    original = pilot.relight_validation_image

    def counted(image):
        relight_calls.append(1)
        return original(image)

    monkeypatch.setattr(pilot, "relight_validation_image", counted)
    report = {}
    pilot.execute(runs, tmp_path, report)
    assert report["maps_written"] == 6
    assert len(relight_calls) == 6
    assert len(calls) == 6 * (1 + len(RELIGHT_NAMES))
    written = sorted(path.name for path in (tmp_path / "maps").iterdir())
    assert written == sorted(f"{name}.npz" for name in pilot.CAN_RUN_NAMES)
    for name in pilot.CAN_RUN_NAMES:
        payload = np.load(tmp_path / "maps" / f"{name}.npz")
        assert payload["category"].item() == "can"
        assert set(RELIGHT_NAMES).issubset(payload.files)
        assert payload["regular"].shape == (1, 4, 4)


def test_a_partial_matrix_does_not_score(tmp_path):
    runs, calls = make_runs()
    report = {}
    with pytest.raises(pilot.PilotFailure, match="all six"):
        pilot.execute(runs[:5], tmp_path, report)
    assert calls == []
    assert not (tmp_path / "regular-checks.json").exists()
    assert list(tmp_path.glob("**/*.npz")) == []


def test_relight_failure_publishes_no_maps(tmp_path, monkeypatch):
    runs, _calls = make_runs()
    original = pilot.relight_validation_image
    seen = []

    def fail_on_last(image):
        seen.append(1)
        if len(seen) == 6:
            raise RuntimeError("synthetic relight failure")
        return original(image)

    monkeypatch.setattr(pilot, "relight_validation_image", fail_on_last)
    with pytest.raises(RuntimeError, match="synthetic relight failure"):
        pilot.execute(runs, tmp_path, {})
    assert not (tmp_path / "maps").exists()
    assert list(tmp_path.glob("**/*.npz")) == []


def test_output_must_be_empty_and_outside_the_heads(tmp_path):
    heads = tmp_path / "heads"
    heads.mkdir()
    with pytest.raises(pilot.PilotFailure, match="overlaps"):
        pilot.assert_fresh_output(heads / "out", heads)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(pilot.PilotFailure, match="not empty"):
        pilot.assert_fresh_output(occupied, heads)
    pilot.assert_fresh_output(tmp_path / "new-out", heads)
