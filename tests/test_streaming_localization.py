import numpy as np
import pytest

pytest.importorskip("scipy")

from lc_simplenet.localization import localization_metrics  # noqa: E402
from lc_simplenet.streaming_localization import NativeMetrics  # noqa: E402


@pytest.mark.parametrize("seed", [0, 8, 21])
@pytest.mark.parametrize("tied", [True, False])
def test_streaming_histogram_matches_sort_reference_across_images(seed, tied):
    rng = np.random.default_rng(seed)
    masks = [rng.integers(0, 2, (5, 7)) for _ in range(3)]
    maps = [
        rng.integers(-2, 3, (5, 7)).astype(np.float32)
        if tied
        else rng.normal(size=(5, 7)).astype(np.float32)
        for _ in masks
    ]
    reference = localization_metrics(maps, masks)
    metrics = NativeMetrics()
    for scores, truth in zip(maps, masks, strict=True):
        metrics.add(scores, truth)
    actual = metrics.result()
    for key in ("regions", "normal_pixels", "defect_pixels", "images"):
        assert actual[key] == reference[key]
    for key in ("pixel_auroc", "au_pro"):
        assert actual[key] == pytest.approx(reference[key], abs=1e-12)


def test_signed_zero_is_one_score_tie():
    scores = np.array([[-0.0, 0.0], [0.0, -0.0]], dtype=np.float32)
    truth = np.array([[0, 0], [1, 1]])
    metrics = NativeMetrics()
    metrics.add(scores, truth)
    assert metrics.result()["pixel_auroc"] == 0.5
    assert metrics.result()["au_pro"] == pytest.approx(0.025)


def test_budget_failure_rejects_image_before_mutating_histograms():
    metrics = NativeMetrics(max_histogram_bytes=26)
    with pytest.raises(MemoryError):
        metrics.add(np.array([[-1.0, 1.0]], dtype=np.float32), np.array([[0, 1]]))
    assert not metrics.blocks and metrics.images == 0


def test_double_precision_scores_are_not_silently_quantized():
    with pytest.raises(ValueError, match="float32"):
        NativeMetrics().add(np.zeros((2, 2)), np.zeros((2, 2)))


def test_float32_boundaries_subnormals_and_image_order_match_reference():
    values = np.array(
        [
            -np.finfo(np.float32).max,
            -1,
            np.nextafter(np.float32(-1), np.float32(0)),
            -np.finfo(np.float32).smallest_subnormal,
            -0.0,
            0.0,
            np.finfo(np.float32).smallest_subnormal,
            np.nextafter(np.float32(1), np.float32(0)),
            1,
            np.nextafter(np.float32(1), np.float32(2)),
            np.finfo(np.float32).max,
            1,
        ],
        dtype=np.float32,
    ).reshape(3, 4)
    truth = np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
    reference = localization_metrics([values, values[::-1]], [truth, truth[::-1]])
    metrics = NativeMetrics()
    metrics.add(values[::-1], truth[::-1])
    metrics.add(values, truth)
    result = metrics.result()
    for key in ("pixel_auroc", "au_pro"):
        assert result[key] == pytest.approx(reference[key], abs=1e-12)
