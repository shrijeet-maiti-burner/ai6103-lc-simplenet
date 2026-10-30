import numpy as np
import pytest

from lc_simplenet.pairs import make_pair, sample_pair


def test_identity_copies_without_aliasing_or_mutating_input():
    image = np.linspace(0, 1, 60, dtype=np.float32).reshape(4, 5, 3)
    pair = make_pair(image, brightness=1, gamma=1)
    np.testing.assert_array_equal(pair.original, image)
    np.testing.assert_array_equal(pair.relit, image)
    assert not np.shares_memory(pair.original, image)
    assert not np.shares_memory(pair.relit, image)
    assert pair.clipped_fraction == 0


def test_relighting_preserves_spatial_marker_and_records_clipping():
    image = np.zeros((4, 5, 3), dtype=np.float32)
    image[1, 3] = [0.25, 0.75, 1.0]
    pair = make_pair(image, brightness=2, gamma=1)
    np.testing.assert_allclose(pair.relit[1, 3], [0.5, 1, 1])
    np.testing.assert_array_equal(pair.relit > 0, image > 0)
    assert pair.clipped_fraction == pytest.approx(2 / image.size)
    assert pair.brightness == 2 and pair.gamma == 1
    assert pair.relit.dtype == image.dtype


def test_matched_rng_produces_identical_b1_b2_pairs_and_logs():
    image = np.full((4, 5, 3), 0.4, dtype=np.float32)
    kwargs = {"brightness_range": (0.8, 1.2), "gamma_range": (0.8, 1.2)}
    b1 = sample_pair(image, rng=np.random.default_rng(7), **kwargs)
    b2 = sample_pair(image, rng=np.random.default_rng(7), **kwargs)
    np.testing.assert_array_equal(b1.relit, b2.relit)
    assert (b1.brightness, b1.gamma) == (b2.brightness, b2.gamma)
    assert 0.8 <= b1.brightness <= 1.2 and 0.8 <= b1.gamma <= 1.2


@pytest.mark.parametrize("bounds", [(0, 1), (2, 1), (1, np.inf), (np.nan, 1)])
def test_invalid_ranges_are_rejected(bounds):
    with pytest.raises(ValueError):
        sample_pair(
            np.zeros((2, 2, 3), dtype=np.float32),
            rng=np.random.default_rng(0),
            brightness_range=bounds,
            gamma_range=(1, 1),
        )
