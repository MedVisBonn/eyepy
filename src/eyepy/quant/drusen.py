from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eyepy.core.annotations import EyeVolumeLayerAnnotation
from eyepy.core.filter import filter_by_height_enface
from eyepy.core.utils import mask_from_boundaries

NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]


def ideal_rpe(rpe_height: NDArrayFloat, bm_height: NDArrayFloat,
              volume_shape: tuple[int, int, int]) -> NDArrayFloat:
    """Compute the ideal RPE from an RPE with Drusen.

    Args:
        rpe_height: The RPE height as offset from the lower border of the B-Scan
        bm_height: The BM height as offset from the lower border of the B-Scan
        volume_shape: Shape of the OCT volume (number of B-Scans, height, width)

    Returns:
        The ideal RPE height as offset from the lower border of the B-Scan
    """
    d, h, w = volume_shape

    # compute shift needed to align the BM to the horizontal center line
    shift = np.empty((d, w), dtype='int')
    shift.fill(h - (h / 2))
    shift = shift - bm_height

    # now shift the RPE location array as well
    shifted_rpe_height = rpe_height + shift

    # Remove all NANs from the shifted RPE data
    clean_shifted = shifted_rpe_height[~np.isnan(shifted_rpe_height)]

    # Compute a histogram with a bin for every pixel height in a B-Scan
    hist, edges = np.histogram(clean_shifted.flatten(),
                               bins=np.arange(volume_shape[1]))

    # Compute the ideal RPE as the mean of the biggest bin and its neighbours
    lower_edge = edges[np.argmax(hist) - 1]
    upper_edge = edges[np.argmax(hist) + 2]
    irpe_height = np.mean(clean_shifted[np.logical_and(
        clean_shifted <= upper_edge, clean_shifted >= lower_edge)])
    ideal = np.full_like(shifted_rpe_height, irpe_height)

    # Shift back into original image space
    ideal = np.reshape(ideal, (d, w)) - shift

    return ideal


def drusen(rpe_height: NDArrayFloat,
           bm_height: NDArrayFloat,
           volume_shape: tuple[int, int, int],
           minimum_height: int = 2) -> NDArrayBool:
    """Compute drusen from the RPE and BM layer segmentation.

    First estimate the ideal RPE based on a histogram of the RPE heights relativ
    to the BM. Then compute drusen as the area between the RPE and the normal RPE

    Args:
        rpe_height: The RPE height as offset from the lower border of the B-Scan
        bm_height: The BM height as offset from the lower border of the B-Scan
        volume_shape: Shape of the OCT volume (number of B-Scans, height, width)
        minimum_height: Minimum height of a drusen in pixels

    Returns:
        A boolean array with the same shape as the OCT volume. True indicates a
        voxel beeing part of a drusen.
    """
    # Estimate ideal RPE
    if isinstance(rpe_height, EyeVolumeLayerAnnotation):
        rpe_height = np.copy(rpe_height.data)
    if isinstance(bm_height, EyeVolumeLayerAnnotation):
        bm_height = np.copy(bm_height.data)

    irpe = ideal_rpe(rpe_height, bm_height, volume_shape)
    # Create drusen map, exclude layer boundaries from the mask (rpe_height+1).
    drusen_map = mask_from_boundaries(
        upper=rpe_height+1, lower=irpe, height=volume_shape[1]
    )
    drusen_map = filter_by_height_enface(drusen_map, minimum_height)

    return drusen_map
