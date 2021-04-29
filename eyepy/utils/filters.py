# -*- coding: utf-8 -*-
import logging
from functools import lru_cache
from typing import Tuple, Union

import numpy as np

Shape = Union[int, Tuple[int, int]]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def filtergrid(
    size: Shape, quadrant_shift: bool = True, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates grid for constructing frequency domain filters.
    Parameters
    ----------
    size : Size of the filter
    quadrant_shift : Quadrant shift such that 0 values / frequencies are at the corners
    normalize: Normalize the range to [-0.5,0.5]
    Returns
    -------
    Coordinate matrices for x and y value for a 2D array. The out can be quadrant shifted
    and / or normalized. This is basically a wrapper around np.meshgrid.

    Inspired by filtergrid.m found at https://www.peterkovesi.com/matlabfns/
    """
    if type(size) is int:
        rows = cols = size
    else:
        rows = size[0]
        cols = size[1]

    range_1 = np.linspace(-(cols // 2), np.floor((cols - 1) / 2), cols)
    range_2 = np.linspace(-(rows // 2), np.floor((rows - 1) / 2), rows)

    if normalize:
        range_1 = range_1 / cols
        range_2 = range_2 / rows

    x, y = np.meshgrid(range_1, range_2)

    # Quadrant shift so that filters are constructed with 0 frequency at the corners
    if quadrant_shift:
        x = np.fft.ifftshift(x)
        y = np.fft.ifftshift(y)

    return x.T, y.T


@lru_cache(maxsize=8)
def radius_filtergrid(
    size: Shape, quadrant_shift: bool = True, normalize: bool = True
) -> np.ndarray:
    """
    Parameters
    ----------
    size : Size of the filter
    quadrant_shift : Quadrant shift such that 0 values / frequencies are at the corners
    normalize: Normalize radius to [0 ,0.5]
    Returns
    -------
    A matrix containing the radius from the center. This radius is in range [0, 0.5] if normalized.
    The result can be quadrant shifted such that the 0 values are in the corners.
    """
    x, y = filtergrid(size, quadrant_shift, normalize)
    radius = np.sqrt(x ** 2 + y ** 2)
    return radius


@lru_cache(maxsize=8)
def theta_filtergrid(size: Shape, quadrant_shift: bool = True) -> np.ndarray:
    """
    Parameters
    ----------
    size : Size of the filter
    quadrant_shift : Quadrant shift such that 0 values / frequencies are at the corners
    Returns
    -------
    A matrix containing the polar angle in radian at the respective position for a circle centered in the matrix.
    The result can be returned quadrant shifted. The angle is 0 for all points on the positive x-axis.
    The angles are pi/2 (90°) and -pi/2 (-90°) on the positive and negative y-axis respectively. On the negative
    x-axis the angle is pi (180°). If you need the angle to be in range [0, 2pi] instead of [-pi, pi], you can simply
    add 2pi whenever the angle is negative.
    """

    y, x = filtergrid(size, quadrant_shift)

    # Matrix values contain polar angle.
    # 0 angle starts on the horizontal line and runs counter clock-wise
    theta = np.arctan2(-y, x)

    return theta
