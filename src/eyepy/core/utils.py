from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skimage.util import img_as_float32
from skimage.util import img_as_ubyte

from eyepy.core.filter import filter_by_height_enface

from .annotations import EyeVolumeLayerAnnotation

NDArrayFloat = npt.NDArray[np.float_]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayInt = npt.NDArray[np.int_]


class DynamicDefaultDict(dict):
    """A defaultdict for which the factory function has access to the missing
    key."""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


def vol_intensity_transform(data: NDArrayFloat) -> NDArrayInt:
    """Wrapper around from_vol_intensity.

    Transform intensities from Heyex VOL exports to achieve a constrast similar to the one used in Heyex.

    Args:
        data: Input data

    Returns:
        Transformed data
    """
    return from_vol_intensity(data)


def from_vol_intensity(data: NDArrayFloat) -> NDArrayInt:
    selection_0 = data == np.finfo(np.float32).max
    selection_data = data <= 1

    new = np.log(data[selection_data] + 2.44e-04)
    new = (new + 8.3) / 8.285

    data[selection_data] = new
    data[selection_0] = 0
    data = np.clip(data, 0, 1)
    return img_as_ubyte(data)


# Function expects numpy array of uint8 type hint
def to_vol_intensity(data: np.ndarray) -> NDArrayFloat:
    data = img_as_float32(data)
    data = data * 8.285 - 8.3
    data = np.exp(data) - 2.44e-04
    return data


def default_intensity_transform(data: np.ndarray) -> np.ndarray:
    """Default intensity transform.

    By default intensities are not changed.

    Args:
        data: Input data

    Returns:
        Input data unchanged
    """
    return data


intensity_transforms = {
    'default': default_intensity_transform,
    'vol': vol_intensity_transform,
}


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
    ideal_rpe = np.full_like(shifted_rpe_height, irpe_height)

    # Shift back into original image space
    ideal_rpe = np.reshape(ideal_rpe, (d, w)) - shift

    return ideal_rpe


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
    # Create drusen map
    drusen_map = np.zeros(volume_shape, dtype=bool)
    # Exclude normal RPE and RPE from the drusen area.
    nans = np.isnan(rpe_height + irpe)

    rpe = np.rint(rpe_height + 1)
    rpe[nans] = 0
    rpe = rpe.astype(int)

    irpe = np.rint(irpe)
    irpe[nans] = 0
    irpe = irpe.astype(int)

    for sli in range(drusen_map.shape[0]):
        for col in range(drusen_map.shape[2]):
            if not nans[sli, col]:
                drusen_map[sli, rpe[sli, col]:irpe[sli, col], col] = 1

    drusen_map = filter_by_height_enface(drusen_map, minimum_height)

    return drusen_map
