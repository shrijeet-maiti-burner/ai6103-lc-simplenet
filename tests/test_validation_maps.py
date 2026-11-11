"""Validation-map generation without a dataset, a head, or a test directory."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lc_simplenet.pilot import PilotConfig
from lc_simplenet.validation_maps import (
    REGULAR_ATOL,
    assert_formal_study_head,
    assert_output_outside_run,
    check_regular_reproduction,
    generate_validation_maps,
    save_validation_maps,
)


def load_generator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_validation_score_maps.py"
    spec = importlib.util.spec_from_file_location("generate_validation_score_maps_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**overrides) -> PilotConfig:
    values = {
        "dataset": "mvtec_ad2",
        "category": "can",
        "manifest_sha256": "a" * 64,
        "variant": "B0",
        "lambda_score": 0.0,
        "image_size": (256, 256),
    }
    values.update(overrides)
    return PilotConfig(**values)


def test_formal_heads_are_b0_b1_at_256_only():
    assert_formal_study_head(_config())
    assert_formal_study_head(_config(variant="B1"))
    with pytest.raises(ValueError, match="B0 and B1"):
        assert_formal_study_head(_config(variant="B2", lambda_score=1.0))
    with pytest.raises(ValueError, match="256"):
        assert_formal_study_head(_config(image_size=(288, 288)))


def test_regular_reproduction_uses_the_absolute_1e3_guard():
    maps = np.zeros((1, 4, 4), dtype=np.float32)
    check_regular_reproduction([REGULAR_ATOL], maps)
    with pytest.raises(ValueError, match="1e-3"):
        check_regular_reproduction([REGULAR_ATOL + 1e-6], maps)


def test_reproduction_failure_does_not_score_relights():
    calls = []

    def predict(images):
        calls.append(np.asarray(images[0]).shape)
        return np.zeros((1, 32, 32), dtype=np.float32)

    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    rows = [{"image_id": "can/validation/good/000.png", "label": 0, "score": 1.0}]
    with pytest.raises(ValueError, match="1e-3"):
        generate_validation_maps("can", 0, "B0", rows, [image], predict)
    assert len(calls) == 1


def test_a_matching_regular_score_scores_all_seven_relights():
    calls = []

    def predict(images):
        calls.append(1)
        return np.full((1, 32, 32), 0.25, dtype=np.float32)

    image = np.full((8, 8, 3), 0.4, dtype=np.float32)
    rows = [{"image_id": "can/validation/good/000.png", "label": 0, "score": 0.25}]
    payload = generate_validation_maps("can", 0, "B1", rows, [image], predict)
    assert len(calls) == 8
    assert payload["regular"].shape == (1, 32, 32)
    assert set(payload["relit"]) == {
        "corner_0.8_0.8",
        "corner_0.8_1.2",
        "corner_1.2_0.8",
        "corner_1.2_1.2",
        "horizontal",
        "vertical",
        "spotlight",
    }
    assert all(view.shape == payload["regular"].shape for view in payload["relit"].values())


def test_test_ids_are_rejected_before_prediction():
    def predict(images):
        raise AssertionError("predict must not run")

    rows = [{"image_id": "can/test_public/good/000.png", "label": 0, "score": 0.0}]
    image = np.full((8, 8, 3), 0.4, dtype=np.float32)
    with pytest.raises(ValueError, match="test"):
        generate_validation_maps("can", 0, "B0", rows, [image], predict)


def test_output_cannot_replace_a_released_run(tmp_path):
    run = tmp_path / "paired" / "can-B0-seed0"
    run.mkdir(parents=True)
    released = run / "heads.pt"
    released.write_bytes(b"released-head")
    with pytest.raises(ValueError, match="released run"):
        assert_output_outside_run(run, run / "maps.npz")
    outside = tmp_path / "maps" / "can-B0-seed0.npz"
    payload = {
        "category": "can",
        "seed": 0,
        "variant": "B0",
        "image_ids": ["can/validation/good/000.png"],
        "regular": np.zeros((1, 8, 8), dtype=np.float32),
        "relit": {
            name: np.zeros((1, 8, 8), dtype=np.float32)
            for name in (
                "corner_0.8_0.8",
                "corner_0.8_1.2",
                "corner_1.2_0.8",
                "corner_1.2_1.2",
                "horizontal",
                "vertical",
                "spotlight",
            )
        },
    }
    save_validation_maps(outside, payload)
    with pytest.raises(FileExistsError):
        save_validation_maps(outside, payload)
    assert released.read_bytes() == b"released-head"
    with pytest.raises(ValueError, match="test"):
        save_validation_maps(tmp_path / "test_public" / "maps.npz", payload)


def test_reader_opens_only_the_validation_file(tmp_path):
    good = tmp_path / "can" / "validation" / "good"
    good.mkdir(parents=True)
    Image.new("RGB", (4, 6), (10, 20, 30)).save(good / "000.png")
    hidden = tmp_path / "can" / "test_public" / "good"
    hidden.mkdir(parents=True)
    Image.new("RGB", (4, 6), (1, 2, 3)).save(hidden / "000.png")
    read = load_generator().read_validation_image
    image = read(tmp_path, "can/validation/good/000.png")
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.float32
    with pytest.raises(ValueError, match="test"):
        read(tmp_path, "can/test_public/good/000.png")
