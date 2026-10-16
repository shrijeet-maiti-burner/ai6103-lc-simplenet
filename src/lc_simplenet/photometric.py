"""Aligned brightness/gamma transform in unnormalized RGB [0, 1]."""

import math

import numpy as np
from numpy.typing import NDArray


def relight(
    image: NDArray[np.floating], *, brightness: float, gamma: float
) -> NDArray[np.floating]:
    """Apply clip(brightness * image**gamma, 0, 1), without spatial operations.

    Call after shared geometry and before ImageNet normalization. This convention
    makes gamma > 1 darker. Parameter selection and visual inspection are required.
    """
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty HWC RGB image")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Expected floating point pixels in [0, 1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite pixels in [0, 1]")
    if not all(math.isfinite(p) and p > 0 for p in (brightness, gamma)):
        raise ValueError("Brightness and gamma must be finite and positive")
    return np.clip(brightness * np.power(image, gamma), 0, 1).astype(image.dtype)
