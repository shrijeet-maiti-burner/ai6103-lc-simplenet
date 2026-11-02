import numpy as np
import pytest

pytest.importorskip("scipy")

from lc_simplenet.localization import (  # noqa: E402
    integrate_to,
    localization_metrics,
    restore_native,
)


def test_perfect_reversed_and_constant_maps_at_five_percent_fpr():
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[1:3, 1:3] = 1
    perfect = localization_metrics([truth.astype(float)], [truth])
    assert perfect["pixel_auroc"] == perfect["au_pro"] == 1
    reversed_result = localization_metrics([1.0 - truth], [truth])
    assert reversed_result["pixel_auroc"] == reversed_result["au_pro"] == 0
    constant = localization_metrics([np.zeros((4, 4))], [truth])
    assert constant["pixel_auroc"] == 0.5
    assert constant["au_pro"] == pytest.approx(0.025)


def test_regions_have_equal_weight_and_use_eight_connectivity():
    truth = np.zeros((6, 6), dtype=np.uint8)
    truth[0, 0] = truth[1, 1] = 1  # one diagonal component of size two
    truth[4, 3:6] = 1  # another component of size three
    scores = np.ones((6, 6))
    scores[truth == 1] = 0
    scores[0, 0] = scores[1, 1] = 2
    result = localization_metrics([scores], [truth])
    assert result["regions"] == 2
    assert result["au_pro"] == pytest.approx(0.5)


def test_ties_are_grouped_and_image_order_does_not_change_metrics():
    rng = np.random.default_rng(21)
    truth = [rng.integers(0, 2, (4, 5)) for _ in range(3)]
    maps = [rng.integers(0, 4, (4, 5)).astype(float) for _ in range(3)]
    first = localization_metrics(maps, truth)
    second = localization_metrics(maps[::-1], truth[::-1])
    for key in ("pixel_auroc", "au_pro"):
        assert first[key] == pytest.approx(second[key], abs=1e-12)


def test_absent_regions_are_unavailable_instead_of_zero():
    result = localization_metrics([np.zeros((3, 3))], [np.zeros((3, 3))])
    assert result["pixel_auroc"] is None and result["au_pro"] is None


def test_interpolated_limit_and_vertical_segments_match_trapezoids():
    assert integrate_to([0, 0, 0.1, 1], [0, 0.5, 1, 1], 0.05) == pytest.approx(0.03125)
    assert integrate_to([0, 0.05, 0.05, 1], [0, 0.3, 0.8, 1], 0.05) == pytest.approx(0.0075)


def test_restoration_keeps_full_frame_geometry_and_rejects_crops():
    ramp = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    actual = restore_native(ramp, [4, 9], map_coordinates="full_frame", crop_box_xyxy=None)
    assert actual.shape == (4, 9) and actual.dtype == np.float32
    assert actual[0, 0] == 0 and actual[-1, -1] == 5
    with pytest.raises(ValueError, match="cropped"):
        restore_native(ramp, [4, 9], map_coordinates="model_input_crop", crop_box_xyxy=[1, 2, 3, 4])


@pytest.mark.parametrize("fault", ["nan", "shape", "label", "limit"])
def test_invalid_maps_and_limits_fail(fault):
    scores, truth = np.zeros((3, 3)), np.zeros((3, 3))
    limit = 0.05
    if fault == "nan":
        scores[0, 0] = np.nan
    if fault == "shape":
        truth = truth[:2]
    if fault == "label":
        truth[0, 0] = 2
    if fault == "limit":
        limit = 0
    with pytest.raises(ValueError):
        localization_metrics([scores], [truth], limit)
