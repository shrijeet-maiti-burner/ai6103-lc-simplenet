"""Native-coordinate localization metrics; no test-fitted scaling or threshold grid."""

import math

import numpy as np
from PIL import Image

METRIC_VERSION = "native-public-1"


def restore_native(scores, original_size, *, map_coordinates, crop_box_xyxy):
    """Restore a full-frame continuous map; interpolation does not recover lost detail."""
    values = np.asarray(scores)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("expected a finite nonempty 2D anomaly map")
    if map_coordinates != "full_frame" or crop_box_xyxy is not None:
        raise ValueError("a cropped map cannot be stretched over the native image")
    if len(original_size) != 2 or any(type(n) is not int or n < 1 for n in original_size):
        raise ValueError("original_size must contain positive integer height and width")
    height, width = original_size
    return np.asarray(
        Image.fromarray(values.astype(np.float32)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
    )


def integrate_to(x, y, limit):
    """Trapezoidal integral, including linear interpolation at the requested FPR limit."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(np.diff(x) < 0)
        or not x[0] <= limit <= x[-1]
    ):
        raise ValueError("expected finite aligned curves sorted by FPR and a covered limit")
    stop = int(np.searchsorted(x, limit, side="right"))
    area = np.sum((y[1:stop] + y[: stop - 1]) * np.diff(x[:stop]) / 2)
    if stop < len(x) and x[stop - 1] < limit:
        value = y[stop - 1] + (y[stop] - y[stop - 1]) * (
            (limit - x[stop - 1]) / (x[stop] - x[stop - 1])
        )
        area += (value + y[stop - 1]) * (limit - x[stop - 1]) / 2
    return float(area)


def localization_metrics(anomaly_maps, ground_truth_maps, fpr_limit=0.05):
    """Equal-weight 8-connected regions, exact score ties, normalized AU-PRO.

    Uses one sort for pixel AUROC and PRO. FPR accumulates in float64; the
    downloaded MVTec v1.0 utility uses float32 FPR, so parity is numerical,
    not a promise of identical bytes. Missing classes/regions yield None.
    """
    from scipy.ndimage import label

    if not math.isfinite(fpr_limit) or not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be finite and in (0,1]")
    if not len(anomaly_maps) or len(anomaly_maps) != len(ground_truth_maps):
        raise ValueError("expected aligned nonempty image and ground-truth lists")
    scores, background, changes = [], [], []
    regions = negatives = positives = 0
    for anomaly, truth in zip(anomaly_maps, ground_truth_maps, strict=True):
        anomaly, truth = np.asarray(anomaly), np.asarray(truth)
        if (
            anomaly.ndim != 2
            or anomaly.shape != truth.shape
            or not anomaly.size
            or not np.isfinite(anomaly).all()
            or not np.isin(truth, [0, 1]).all()
        ):
            raise ValueError("maps must be aligned finite 2D scores and binary native masks")
        connected, count = label(truth, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(connected.ravel())
        weights = np.zeros(len(sizes), dtype=np.float64)
        weights[1:] = 1.0 / sizes[1:]
        scores.append(anomaly.ravel())
        background.append((truth == 0).ravel())
        changes.append(weights[connected].ravel())
        regions += count
        negatives += int(sizes[0])
        positives += int(truth.size - sizes[0])
    result = {
        "metric_version": METRIC_VERSION,
        "images": len(scores),
        "normal_pixels": negatives,
        "defect_pixels": positives,
        "regions": regions,
        "connectivity": 8,
        "fpr_limit": fpr_limit,
        "pixel_auroc": None,
        "au_pro": None,
        "normalization": "trapezoid integral to FPR limit, divided by that limit",
    }
    if not negatives or not positives or not regions:
        result["status"] = "unavailable: missing normal pixels or defect regions"
        return result
    values = np.concatenate(scores)
    del scores
    order = np.argsort(values, kind="stable")[::-1]
    sorted_scores = values[order]
    del values
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    del sorted_scores
    fp = np.concatenate(background)[order]
    del background
    fp = np.cumsum(fp, dtype=np.uint64)[ends]
    pro = np.concatenate(changes)[order]
    del changes, order
    pro = np.cumsum(pro, dtype=np.float64)[ends] / regions
    fpr = np.r_[0.0, fp / negatives, 1.0]
    tpr = np.r_[0.0, (ends + 1 - fp) / positives, 1.0]
    pro = np.r_[0.0, np.clip(pro, 0, 1), 1.0]
    result.update(
        pixel_auroc=integrate_to(fpr, tpr, 1.0),
        au_pro=integrate_to(fpr, pro, fpr_limit) / fpr_limit,
        status="complete",
    )
    return result
