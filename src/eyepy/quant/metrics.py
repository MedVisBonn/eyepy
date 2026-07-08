"""Core metric calculations for quantification.

This module provides fundamental metric calculations for area, distance,
and other quantitative measures in ophthalmic images.
"""

import numpy as np
import numpy.typing as npt


def compute_area(
    mask: npt.NDArray[np.bool_],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> float:
    """Compute area of a binary mask in physical units.

    Args:
        mask: Binary mask (True = region of interest)
        scale_x: Micrometers per pixel in x-direction (default: 1.0)
        scale_y: Micrometers per pixel in y-direction (default: 1.0)

    Returns:
        Area in square micrometers (or square pixels if scales are 1.0)
    """
    pixel_area = scale_x * scale_y
    n_pixels = np.sum(mask)
    return float(n_pixels * pixel_area)


def masked_mean(
    data: npt.NDArray,
    mask: npt.NDArray[np.bool_],
) -> float:
    """Compute mean of data where mask is True and data is finite.

    Args:
        data: Scalar field.
        mask: Boolean region mask.

    Returns:
        Mean value, or NaN if no valid pixels exist.
    """
    valid = mask & np.isfinite(data)
    if not np.any(valid):
        return float('nan')
    return float(np.nanmean(data[valid]))


def masked_median(
    data: npt.NDArray,
    mask: npt.NDArray[np.bool_],
) -> float:
    """Compute median of data where mask is True and data is finite.

    Args:
        data: Scalar field.
        mask: Boolean region mask.

    Returns:
        Median value, or NaN if no valid pixels exist.
    """
    valid = mask & np.isfinite(data)
    if not np.any(valid):
        return float('nan')
    return float(np.nanmedian(data[valid]))
