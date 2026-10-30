"""Build aligned pixel-space pairs after the caller's shared spatial transform."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lc_simplenet.photometric import relight


@dataclass(frozen=True)
class LightingPair:
    original: NDArray[np.floating]
    relit: NDArray[np.floating]
    brightness: float
    gamma: float
    clipped_fraction: float


def _range(bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Lighting ranges must be finite, positive and ordered")
    return low, high


def make_pair(image: NDArray[np.floating], *, brightness: float, gamma: float) -> LightingPair:
    """Copy HWC RGB [0,1] into aligned views; never crop or normalize here.

    clipped_fraction counts channel values exceeding 1 before clipping, not
    merely existing white pixels. This diagnostic is not a defect-preservation test.
    """
    altered = relight(image, brightness=brightness, gamma=gamma)
    with np.errstate(over="ignore"):
        before_clip = brightness * np.power(image.astype(np.float64), gamma)
    return LightingPair(
        original=image.copy(),
        relit=altered,
        brightness=float(brightness),
        gamma=float(gamma),
        clipped_fraction=float(np.mean(before_clip > 1)),
    )


def sample_pair(
    image: NDArray[np.floating],
    *,
    rng: np.random.Generator,
    brightness_range: tuple[float, float],
    gamma_range: tuple[float, float],
) -> LightingPair:
    """Sample independent uniform parameters using an explicit caller-owned RNG.

    B1 and B2 must call this same function with matching RNG state and ranges.
    No final study ranges are chosen by this helper.
    """
    brightness_bounds = _range(brightness_range)
    gamma_bounds = _range(gamma_range)
    return make_pair(
        image,
        brightness=float(rng.uniform(*brightness_bounds)),
        gamma=float(rng.uniform(*gamma_bounds)),
    )
