import numpy as np
import pytest
from PIL import Image

from lc_simplenet.public_scoring import (
    condition_from_name,
    list_public_test,
    read_resized,
    score_records,
)


def make_tree(root, category="can"):
    layout = {"good": ["000_regular", "001_shift_2"], "bad": ["002_overexposed"]}
    for folder, names in layout.items():
        directory = root / category / "test_public" / folder
        directory.mkdir(parents=True)
        for name in names:
            Image.new("RGB", (8, 4), (10, 20, 30)).save(directory / f"{name}.png")
    (root / category / "test_private").mkdir()
    Image.new("RGB", (8, 4)).save(root / category / "test_private" / "secret.png")


def test_lists_only_public_test_with_folder_labels(tmp_path):
    make_tree(tmp_path)
    images = list_public_test(tmp_path, "can")
    assert [(i.image_id, i.label) for i in images] == [
        ("can/test_public/good/000_regular.png", 0),
        ("can/test_public/good/001_shift_2.png", 0),
        ("can/test_public/bad/002_overexposed.png", 1),
    ]
    assert all("private" not in i.image_id for i in images)


def test_missing_label_folder_is_an_error(tmp_path):
    (tmp_path / "can" / "test_public" / "good").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        list_public_test(tmp_path, "can")


def test_condition_names_are_strict():
    assert condition_from_name("can/test_public/good/001_shift_2.png") == "shift_2"
    assert condition_from_name("can/test_public/bad/002_overexposed.png") == "overexposed"
    with pytest.raises(ValueError):
        condition_from_name("can/test_public/good/002_dim.png")


def test_read_resized_shape_range_and_original_size(tmp_path):
    make_tree(tmp_path)
    pixels, original = read_resized(tmp_path / "can/test_public/good/000_regular.png", (16, 16))
    assert pixels.shape == (16, 16, 3) and pixels.dtype == np.float32
    assert original == [4, 8]
    np.testing.assert_allclose(pixels[0, 0], np.array([10, 20, 30]) / 255, atol=1e-6)


def test_score_is_max_patch_and_rows_carry_geometry(tmp_path):
    make_tree(tmp_path)
    images = list_public_test(tmp_path, "can")

    calls = []

    def predict(batch):
        assert len(batch) == 1 and batch[0].shape == (16, 16, 3)
        patch = np.zeros((1, 2, 2), dtype=np.float32)
        calls.append(1)
        patch[0, 1, 1] = float(len(calls))
        return patch

    rows, maps = score_records(images, predict, (16, 16), (16, 16))
    assert [r["score"] for r in rows] == [1.0, 2.0, 3.0]
    assert [r["label"] for r in rows] == [0, 0, 1]
    assert maps.shape == (3, 2, 2)
    assert rows[0]["original_size"] == [4, 8] and rows[0]["map_coordinates"] == "full_frame"


def test_non_finite_prediction_is_rejected(tmp_path):
    make_tree(tmp_path)
    images = list_public_test(tmp_path, "can")
    with pytest.raises(FloatingPointError):
        score_records(images, lambda b: np.full((1, 2, 2), np.nan), (16, 16), (16, 16))
