import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from lc_simplenet.external_localization import ExternalNativeMetrics
from lc_simplenet.localization import localization_metrics


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("tied", [False, True])
def test_exact_external_parity(tmp_path, seed, tied):
    rng = np.random.default_rng(seed)
    scores = [rng.normal(size=(11, 13)).astype(np.float32) for _ in range(3)]
    truth = [(rng.random((11, 13)) > 0.8).astype(np.uint8) for _ in scores]
    truth[0][:] = 0
    if tied:
        scores = [np.round(s * 2) / 2 for s in scores]
    expected = localization_metrics(scores, truth)
    metrics = ExternalNativeMetrics(tmp_path / "work", prefix_bits=16)
    for s, t in zip(scores, truth, strict=True):
        metrics.add(s, t)
    actual = metrics.result()
    json.dumps(actual, allow_nan=False)
    assert actual["regions"] == expected["regions"]
    assert actual["normal_pixels"] == expected["normal_pixels"]
    assert actual["pixel_auroc"] == pytest.approx(expected["pixel_auroc"], abs=1e-13)
    assert actual["au_pro"] == pytest.approx(expected["au_pro"], abs=1e-13)
    assert metrics.result() == actual
    metrics.cleanup()
    assert not (tmp_path / "work").exists()


def test_float_edges_signed_zero_and_unequal_regions(tmp_path):
    scores = np.array(
        [[-0.0, 0.0, np.nextafter(np.float32(0), np.float32(1))], [-1e30, 1e30, -1e-30]],
        dtype=np.float32,
    )
    truth = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.uint8)
    expected = localization_metrics([scores], [truth])
    metrics = ExternalNativeMetrics(tmp_path / "work", prefix_bits=16)
    metrics.add(scores, truth)
    actual = metrics.result()
    json.dumps(actual, allow_nan=False)
    assert actual["pixel_auroc"] == pytest.approx(expected["pixel_auroc"], abs=1e-14)
    assert actual["au_pro"] == pytest.approx(expected["au_pro"], abs=1e-14)
    metrics.cleanup()


def test_input_budget_and_missing_class(tmp_path):
    with pytest.raises(MemoryError):
        ExternalNativeMetrics(tmp_path / "bad", max_histogram_bytes=100)
    metrics = ExternalNativeMetrics(tmp_path / "work")
    with pytest.raises(ValueError):
        metrics.add(np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2)))
    metrics.add(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2)))
    assert metrics.result()["pixel_auroc"] is None
    metrics.cleanup()


def test_disk_write_failure_invalidates_the_accumulator(tmp_path, monkeypatch):
    metrics = ExternalNativeMetrics(tmp_path / "work")

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(metrics, "_append", fail)
    with pytest.raises(OSError):
        metrics.add(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="invalidated"):
        metrics.result()
    metrics.cleanup()


@pytest.mark.parametrize("prefix_bits", [12, 24])
def test_radix_width_changes_resources_but_not_score_ties(tmp_path, prefix_bits):
    values = np.array(
        [[-1, -0.0, 0.0], [0.5, 1, np.nextafter(np.float32(1), np.float32(2))]], dtype=np.float32
    )
    truth = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.uint8)
    reference = localization_metrics([values], [truth])
    metrics = ExternalNativeMetrics(tmp_path / "work", prefix_bits=prefix_bits)
    metrics.add(values, truth)
    actual = metrics.result()
    assert actual["pixel_auroc"] == pytest.approx(reference["pixel_auroc"], abs=1e-14)
    assert actual["au_pro"] == pytest.approx(reference["au_pro"], abs=1e-14)
    metrics.cleanup()


def test_cleanup_never_removes_an_unrecognized_file(tmp_path):
    work = tmp_path / "work"
    metrics = ExternalNativeMetrics(work)
    metrics.add(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2)))
    metrics.result()
    sentinel = work / "unrelated.txt"
    sentinel.write_text("retain")
    with pytest.raises(OSError):
        metrics.cleanup()
    assert sentinel.read_text() == "retain"
