import numpy as np
import pytest

from lc_simplenet.geometry import restore_score_map


def test_native_identity_preserves_negative_scores_and_input():
    scores = np.arange(35, dtype=np.float32).reshape(5, 7) - 17
    before = scores.copy()
    restored = restore_score_map(scores, original_size=(5, 7), image_size=(5, 7))
    np.testing.assert_array_equal(restored, before)
    np.testing.assert_array_equal(scores, before)
    assert restored.dtype == np.float32


def test_padding_is_removed_before_native_resize_and_keeps_border_signal():
    scores = np.full((8, 8), 999, dtype=np.float32)
    scores[:5, :7] = 0
    scores[4, 6] = 17
    restored = restore_score_map(
        scores, original_size=(10, 14), image_size=(8, 8), valid_size=(5, 7)
    )
    assert restored.shape == (10, 14)
    assert restored[-1, -1] == 17
    assert restored.max() == 17 and restored.min() == 0


def test_coarse_constant_map_keeps_value_through_resizing_and_padding_crop():
    restored = restore_score_map(
        np.full((2, 3), -3.5, dtype=np.float16),
        original_size=(30, 58),
        image_size=(16, 32),
        valid_size=(15, 29),
    )
    assert restored.shape == (30, 58)
    np.testing.assert_array_equal(restored, -3.5)


def test_integer_array_sizes_from_dataset_metadata_are_accepted():
    restored = restore_score_map(
        np.ones((2, 3), dtype=np.float32),
        original_size=np.array([5, 7], dtype=np.int64),
        image_size=np.array([4, 6], dtype=np.int64),
    )
    assert restored.shape == (5, 7)
    np.testing.assert_array_equal(restored, 1)


@pytest.mark.parametrize(
    "scores",
    [np.zeros((2, 2, 1)), np.zeros((2, 2), dtype=np.uint8), np.full((2, 2), np.nan)],
)
def test_invalid_maps_and_discrete_masks_are_rejected(scores):
    with pytest.raises(ValueError, match="2D floating-point"):
        restore_score_map(scores, original_size=(4, 4), image_size=(4, 4))


def test_invalid_valid_region_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        restore_score_map(
            np.zeros((2, 2)), original_size=(4, 4), image_size=(4, 4), valid_size=(5, 4)
        )
