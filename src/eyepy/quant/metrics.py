"""Core metric calculations for quantification.

This module provides fundamental metric calculations for area, distance,
and other quantitative measures in ophthalmic images.
"""

from typing import Optional

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
