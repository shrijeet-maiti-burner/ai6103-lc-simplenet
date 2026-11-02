"""Preview preprocessing must match what the AD 2 model receives."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_lighting.py"


def test_preview_resizes_full_non_square_frame_like_training(tmp_path):
    source = np.random.default_rng(9).integers(0, 256, (43, 79, 3), dtype=np.uint8)
    path = tmp_path / "training.png"
    Image.fromarray(source).save(path)
    module = runpy.run_path(str(SCRIPT))
    actual = module["load_training_preview"](path)
    expected = np.asarray(
        Image.fromarray(source).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / np.float32(255)
    assert actual.shape == (256, 256, 3)
    np.testing.assert_array_equal(actual, expected)


def test_preview_metadata_and_clipping_use_resized_pixels(tmp_path, monkeypatch):
    source = np.zeros((11, 27, 3), dtype=np.uint8)
    source[:, ::2] = 255
    path, output = tmp_path / "training.png", tmp_path / "preview"
    Image.fromarray(source).save(path)
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--image", str(path), "--output-dir", str(output)]
    )
    module["main"]()
    metadata = json.loads((output / "parameters.json").read_text())
    assert metadata["input_size"] == [256, 256]
    assert "full-frame bilinear" in metadata["scope"]
    pixels = module["load_training_preview"](path).astype(np.float64)
    for settings in metadata["settings"]:
        expected = np.mean(settings["brightness"] * pixels ** settings["gamma"] > 1)
        assert settings["clipped_fraction"] == pytest.approx(expected)
    assert (output / "lighting_preview.png").is_file()
