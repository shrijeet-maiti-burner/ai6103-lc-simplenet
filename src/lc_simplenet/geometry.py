"""Restore continuous score maps after full-image resizing and bottom/right padding."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image


def _size(value: ArrayLike, name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,) or not np.issubdtype(array.dtype, np.integer) or (array <= 0).any():
        raise ValueError(f"{name} must be positive integer (height, width)")
    return int(array[0]), int(array[1])


def restore_score_map(
    scores: NDArray[np.floating],
    *,
    original_size: ArrayLike,
    image_size: ArrayLike,
    valid_size: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Upsample to model-input size, crop padding, then resize to native size.

    ``valid_size`` describes the top-left valid image before bottom/right
    padding. Omit it when there was no padding. Interpolation is PIL bilinear
    in float32, without score normalization or clipping. Discrete masks, crops,
    rotations and other spatial transforms require a separate geometry rule.
    """
    native_h, native_w = _size(original_size, "original_size")
    input_h, input_w = _size(image_size, "image_size")
    valid_h, valid_w = _size(valid_size if valid_size is not None else image_size, "valid_size")
    if valid_h > input_h or valid_w > input_w:
        raise ValueError("valid_size exceeds image_size")
    if (
        scores.ndim != 2
        or not scores.size
        or not np.issubdtype(scores.dtype, np.floating)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Expected a finite, nonempty 2D floating-point score map")
    with np.errstate(over="ignore", invalid="ignore"):
        array = scores.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Score values exceed float32 range")
    image = Image.fromarray(array)
    if image.size != (input_w, input_h):
        image = image.resize((input_w, input_h), Image.Resampling.BILINEAR)
    image = image.crop((0, 0, valid_w, valid_h))
    if image.size != (native_w, native_h):
        image = image.resize((native_w, native_h), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.float32)
