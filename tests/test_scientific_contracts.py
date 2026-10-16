import numpy as np
import pytest

from lc_simplenet.metrics import image_counts, normal_validation_threshold
from lc_simplenet.photometric import relight


def test_relighting_keeps_marker_positions_and_input():
    image = np.zeros((3, 4, 3), dtype=np.float32)
    image[1, 2, :] = 0.5
    source = image.copy()
    altered = relight(image, brightness=0.8, gamma=2.0)
    np.testing.assert_array_equal(image, source)
    np.testing.assert_allclose(altered[1, 2], 0.2)
    np.testing.assert_array_equal(altered > 0, source > 0)
    assert altered.dtype == image.dtype


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.full((2, 2, 3), np.nan)])
def test_relighting_rejects_invalid_images(bad):
    with pytest.raises(ValueError):
        relight(bad, brightness=1.0, gamma=1.0)


def test_frozen_threshold_and_missing_denominators():
    threshold = normal_validation_threshold([0, 1, 2, 3, 4], target_fpr=0.25)
    assert threshold == 3
    regular = image_counts([1, 3, 4, 5], [0, 0, 1, 1], threshold)
    shifted = image_counts([4, 5, 2, 6], [0, 0, 1, 1], threshold)
    assert regular["fpr"] == 0 and regular["recall"] == 1
    assert shifted["fpr"] == 1 and shifted["recall"] == 0.5
    assert shifted["false_positives"] == shifted["normal_images"] == 2
    assert image_counts([0, 1], [0, 0], threshold)["recall"] is None


def test_tied_calibration_and_unknown_labels():
    threshold = normal_validation_threshold([1, 1, 1], 0.05)
    assert image_counts([1, 1], [0, 0], threshold)["false_positives"] == 0
    with pytest.raises(ValueError):
        image_counts([0.1], [-1], threshold)
    with pytest.raises(ValueError):
        normal_validation_threshold([], 0.05)


def test_mse_matches_analytic_gradient_on_both_branches():
    torch = pytest.importorskip("torch")
    from lc_simplenet.losses import patch_score_consistency

    left = torch.tensor([1.0, 3.0], requires_grad=True)
    right = torch.tensor([2.0, 1.0], requires_grad=True)
    loss = patch_score_consistency(left, right)
    assert loss.item() == 2.5
    loss.backward()
    torch.testing.assert_close(left.grad, torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(right.grad, torch.tensor([1.0, -2.0]))
    with pytest.raises(ValueError):
        patch_score_consistency(left, right[:, None])
