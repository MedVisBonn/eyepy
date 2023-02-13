import numpy as np
from skimage import img_as_float32
from skimage import img_as_ubyte

from eyepy.core.filter import filter_by_height_enface

from .annotations import EyeVolumeLayerAnnotation


class DynamicDefaultDict(dict):
    """A defaultdict for which the factory function has access to the missing key"""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


def vol_intensity_transform(data):
    """ Wrapper around from_vol_intensity might be removed in future

    Args:
        data:

    Returns:

    """
    return from_vol_intensity(data)


def from_vol_intensity(data):
    selection_0 = data == np.finfo(np.float32).max
    selection_data = data <= 1

    new = np.log(data[selection_data] + 2.44e-04)
    new = (new + 8.3) / 8.285

    data[selection_data] = new
    data[selection_0] = 0
    data = np.clip(data, 0, 1)
    return img_as_ubyte(data)


# Function expects numpy array of uint8 type hint
def to_vol_intensity(data):
    data = img_as_float32(data)
    data = data * 8.285 - 8.3
    data = np.exp(data) - 2.44e-04
    return data


def default_intensity_transform(data):
    """

    Args:
        data:

    Returns:

    """
    return data


intensity_transforms = {
    "default": default_intensity_transform,
    "vol": vol_intensity_transform,
}


def ideal_rpe(rpe_height, bm_height, volume_shape):
    """

    Args:
        rpe_height:
        bm_height:
        volume_shape:

    Returns:

    """
    d, h, w = volume_shape

    # compute shift needed to align the BM to the horizontal center line
    shift = np.empty((d, w), dtype="int")
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


def drusen(rpe_height, bm_height, volume_shape, minimum_height=2):
    """Compute drusen from the RPE and BM layer segmentation.

    First estimate the ideal RPE based on a histogram of the RPE heights relativ
    to the BM. Then compute drusen as the area between the RPE and the normal RPE

    Args:
        rpe_height:
        bm_height:
        volume_shape:
        minimum_height:

    Returns:

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
