"""Image-level operating points. These are not AD 2's server metrics."""

import numpy as np
from numpy.typing import ArrayLike


def _scores(values: ArrayLike) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty finite score vector")
    return scores


def normal_validation_threshold(scores: ArrayLike, target_fpr: float) -> float:
    """Higher empirical quantile on normal regular-light validation images only.

    Anomaly decision is strictly score > threshold. This controls the observed
    validation FPR; it does not guarantee a population or shifted-light FPR.
    """
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between 0 and 1")
    return float(np.quantile(_scores(scores), 1 - target_fpr, method="higher"))


def image_counts(scores: ArrayLike, labels: ArrayLike, threshold: float) -> dict:
    """Known image labels only: 0 normal, 1 anomalous. Missing rates are None."""
    values = _scores(scores)
    truth = np.asarray(labels)
    if truth.shape != values.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("Labels must match scores and contain only 0 or 1")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predicted = values > threshold
    normal, defect = truth == 0, truth == 1
    fp, tp = int((predicted & normal).sum()), int((predicted & defect).sum())
    n_normal, n_defect = int(normal.sum()), int(defect.sum())
    return {
        "false_positives": fp,
        "normal_images": n_normal,
        "true_positives": tp,
        "defective_images": n_defect,
        "fpr": fp / n_normal if n_normal else None,
        "recall": tp / n_defect if n_defect else None,
    }
