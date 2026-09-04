"""Enface warping utilities for B-scan maps."""

import numpy as np
import numpy.typing as npt
from skimage import transform


def warp_bscan_to_enface(
    bscan_map: npt.NDArray,
    localizer_transform,
    output_shape: tuple[int, int],
    order: int = 1,
) -> npt.NDArray:
    """Warp a B-scan-space scalar map to the localizer/enface plane.

    Args:
        bscan_map: 2D map in B-scan space, shape (n_bscans, width).
        localizer_transform: Affine transform from volume to localizer.
        output_shape: Output shape (localizer.size_y, localizer.size_x).
        order: Interpolation order (1 for continuous maps, 0 for labels).

    Returns:
        Warped map in enface/localizer coordinates.
    """
    return transform.warp(
        np.flip(bscan_map, axis=0),
        localizer_transform.inverse,
        output_shape=output_shape,
        order=order,
        cval=np.nan,
    )
