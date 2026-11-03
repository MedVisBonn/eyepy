import numpy as np

from eyepy.core.utils import display_to_vol_linear_approx
from eyepy.core.utils import vol_linear_to_display


def test_observed_vol_display_thresholds():
    # Non-identifying intensity/grey pairs from the matched VOL/TIFF exports.
    values = np.array([
        0, 8.054077625274658e-6, 1.6376376152038574e-5, .001708984375,
        .015380859375, 1
    ],
                      dtype=np.float32)
    original = values.copy()
    np.testing.assert_array_equal(vol_linear_to_display(values),
                                  [0, 1, 2, 63, 127, 255])
    np.testing.assert_array_equal(values, original)


def test_vol_display_invalid_values_and_saturation():
    values = np.array(
        [np.finfo(np.float32).max, np.nan, np.inf, -np.inf, -1, 0, 1, 2],
        dtype=np.float32)
    np.testing.assert_array_equal(vol_linear_to_display(values),
                                  [0, 0, 0, 0, 0, 0, 255, 255])


def test_representative_inverse_round_trips_every_grey():
    gray = np.arange(256, dtype=np.uint8)
    linear = display_to_vol_linear_approx(gray)
    assert linear.dtype == np.float32
    assert linear[0] == 0 and linear[-1] == 1
    np.testing.assert_array_equal(vol_linear_to_display(linear), gray)
    np.testing.assert_array_equal(display_to_vol_linear_approx(gray.astype(float) / 255),
                                  linear)


def test_approximate_inverse_cannot_recover_original_linear_values():
    original = np.array([.1, .1001], np.float32)
    display = vol_linear_to_display(original)
    assert display[0] == display[1]
    recovered = display_to_vol_linear_approx(display)
    assert recovered[0] == recovered[1]
    assert not np.array_equal(original, recovered)


def test_legacy_intensity_function_names_remain_available():
    import pytest

    from eyepy.core.utils import from_vol_intensity
    from eyepy.core.utils import to_vol_intensity

    grey = np.array([0, 127, 255], np.uint8)
    with pytest.warns(DeprecationWarning, match='display_to_vol_linear_approx'):
        linear = to_vol_intensity(grey)
    with pytest.warns(DeprecationWarning, match='vol_linear_to_display'):
        np.testing.assert_array_equal(from_vol_intensity(linear), grey)


def test_approximate_inverse_rejects_invalid_display_inputs():
    import pytest

    for values in (np.array([np.nan]), np.array([np.inf]), np.array([1.1]),
                   np.array([-.1]), np.array([127], np.int32)):
        with pytest.raises(ValueError, match='greys'):
            display_to_vol_linear_approx(values)
