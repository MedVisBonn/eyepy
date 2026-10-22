import numpy as np

from eyepy.core.utils import from_vol_intensity
from eyepy.core.utils import to_vol_intensity


def test_observed_vol_display_thresholds():
    # Non-identifying intensity/grey pairs from the matched VOL/TIFF exports.
    values = np.array([
        0, 8.054077625274658e-6, 1.6376376152038574e-5, .001708984375,
        .015380859375, 1
    ],
                      dtype=np.float32)
    original = values.copy()
    np.testing.assert_array_equal(from_vol_intensity(values),
                                  [0, 1, 2, 63, 127, 255])
    np.testing.assert_array_equal(values, original)


def test_vol_display_invalid_values_and_saturation():
    values = np.array(
        [np.finfo(np.float32).max, np.nan, np.inf, -np.inf, -1, 0, 1, 2],
        dtype=np.float32)
    np.testing.assert_array_equal(from_vol_intensity(values),
                                  [0, 0, 0, 0, 0, 0, 255, 255])


def test_representative_inverse_round_trips_every_grey():
    gray = np.arange(256, dtype=np.uint8)
    linear = to_vol_intensity(gray)
    assert linear.dtype == np.float32
    assert linear[0] == 0 and linear[-1] == 1
    np.testing.assert_array_equal(from_vol_intensity(linear), gray)
    np.testing.assert_array_equal(to_vol_intensity(gray.astype(float) / 255),
                                  linear)
