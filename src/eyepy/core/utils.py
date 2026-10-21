from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eyepy.core.annotations import EyeVolumeLayerAnnotation
from eyepy.core.filter import filter_by_height_enface

NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]
NDArrayInt = npt.NDArray[np.int64]
NDArrayUByte = npt.NDArray[np.ubyte]

# Empirical display thresholds, shared by the forward and representative
# inverse mappings. See docs/formats/he_vol.md for provenance and limits.
_HEYEX_LOG_SLOPE = 30.776116889447362
_HEYEX_LOG_OFFSET = 0.001931325892041094


class DynamicDefaultDict(dict):
    """A defaultdict for which the factory function has access to the missing
    key."""

    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


def angio_intensity_transform(data: NDArrayFloat) -> NDArrayInt:
    """Wrapper around from_angio_intensity.

    Transform intensities from Heyex ANGIO exports to achieve a constrast similar to the one used in Heyex.

    Args:
        data: Input data

    Returns:
        Transformed data
    """
    return from_angio_intensity(data)


def from_angio_intensity(data: NDArrayFloat) -> NDArrayInt:
    # Zero anything that is not in the range [0, 1]
    data[np.isnan(data)] = 0
    return np.where((data < 0) | (data > 1), 0, data)


def vol_intensity_transform(data: NDArrayFloat) -> NDArrayUByte:
    """Wrapper around from_vol_intensity.

    Transform intensities from Heyex VOL exports to achieve a constrast similar to the one used in Heyex.

    Args:
        data: Input data

    Returns:
        Transformed data
    """
    return from_vol_intensity(data)


def from_vol_intensity(data: NDArrayFloat) -> NDArrayUByte:
    """Render linear VOL intensities using the empirically verified HEYEX curve.

    The global coefficients were fitted to five default-display VOL/TIFF
    pairs and verified on 218 B-scans. They are not vendor
    specifications. Missing values render black. The input array is
    never modified.
    """
    values = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(values) & (values != np.finfo(np.float32).max)
    values = np.where(valid, np.clip(values, 0, 1), 0)
    gray = np.floor(_HEYEX_LOG_SLOPE * np.log1p(4096 * values) +
                    _HEYEX_LOG_OFFSET)
    return np.clip(gray, 0, 255).astype(np.ubyte)

def from_e2e_intensity(data: NDArrayFloat32) -> NDArrayUByte:
    """Unnormalized fallback for E2E data without registration metadata."""
    return from_vol_intensity(np.where(data == 1.9990234375, 0, data))


# Function expects numpy array of uint8 type hint
def to_vol_intensity(data: np.ndarray) -> NDArrayFloat32:
    """Choose a linear intensity in each display bin.

    Accept uint8 greys or floating-point greys in [0, 1]. Display
    conversion loses information; this cannot recover the original
    linear intensities.
    """
    from skimage.util import img_as_float64

    gray = img_as_float64(data) * 255
    linear = np.expm1((gray + .5 - _HEYEX_LOG_OFFSET) / _HEYEX_LOG_SLOPE) / 4096
    linear = np.where(gray == 0, 0, np.where(gray == 255, 1, linear))
    return np.clip(linear, 0, 1).astype(np.float32)


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
    'e2e': from_e2e_intensity,
    'angio': angio_intensity_transform,
}


def sum_par_algorithm(data: np.ndarray) -> np.ndarray:
    data[np.isnan(data)] = 0
    data = np.where((data < 0) | (data > 1), 0, data)

    csum = np.cumsum(data[:,::-1,:], axis=1)[:, ::-1, :]
    max_val = np.nanmax(csum, axis=1, keepdims=True)
    csum = np.divide(csum, max_val, out=np.zeros_like(csum), where=max_val != 0)

    par_data = data * csum
    max_val = np.nanmax(par_data, axis=1, keepdims=True)
    par_data = np.divide(par_data, max_val, out=np.zeros_like(par_data), where=max_val != 0)
    return par_data


def default_par_algorithm(data: np.ndarray) -> np.ndarray:
    """Default Projection Artifact Removal (PAR) algorithm."""
    return data


par_algorithms = {
    'default': default_par_algorithm,
    'sum': sum_par_algorithm
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
    # Create drusen map, exclude layer boundaries from the mask (rpe_height+1).
    drusen_map = mask_from_boundaries_loop(
        upper=rpe_height+1, lower=irpe, height=volume_shape[1]
    )
    drusen_map = filter_by_height_enface(drusen_map, minimum_height)

    return drusen_map

def mask_from_boundaries(upper: np.ndarray, lower: np.ndarray, height: int) -> np.ndarray:
    """Compute a boolean mask from upper and lower boundaries for each column
    using numpy broadcasting.

    The mask includes the upper boundary and excludes the lower boundary.
    To exclude the upper layer, add 1 to it before calling this function.
    To include the lower layer, add 1 to it before calling this function.

    This method is generally faster and more efficient for typical use cases.
    Use this when the proportion of NaN values in the boundaries is below 20%.
    For large volumes with mostly valid data, broadcasting is recommended.

    Args:
        upper: 2D array (slices, columns) of upper boundary indices.
        lower: 2D array (slices, columns) of lower boundary indices.
        height: int, the number of rows (y-dimension) in the mask.

    Returns:
        mask: 3D boolean array (slices, height, columns)
    """
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    nans = np.isnan(upper) | np.isnan(lower)
    # Avoid casting issues with NaNs
    upper = np.where(nans, -1, upper)
    lower = np.where(nans, -1, lower)
    # Round to nearest integer and clip to valid range
    upper_idx = np.rint(upper).astype(int)
    lower_idx = np.rint(lower).astype(int)
    upper_idx = np.clip(upper_idx, 0, height)
    lower_idx = np.clip(lower_idx, 0, height)

    valid = ~nans & (upper_idx < lower_idx)
    y_coords = np.arange(height)
    mask = (
        (y_coords[None, :, None] >= upper_idx[:, None, :]) &
        (y_coords[None, :, None] < lower_idx[:, None, :]) &
        (valid[:, None, :])
    )
    return mask


def mask_from_boundaries_loop(
    upper: np.ndarray, lower: np.ndarray, height: int
) -> np.ndarray:
    """Looping variant of mask_from_boundaries, iterating only over valid columns.

    The mask includes the upper boundary and excludes the lower boundary.
    To exclude the upper layer, add 1 to it before calling this function.
    To include the lower layer, add 1 to it before calling this function.

    This method can be more efficient than broadcasting when the NaN or invalid
    column proportion (where upper > lower) is above 20%.
    For highly masked data, the looping approach avoids extra work.

    Args:
        upper: 2D array (slices, columns) of upper boundary indices.
        lower: 2D array (slices, columns) of lower boundary indices.
        height: int, the number of rows (y-dimension) in the mask.

    Returns:
        mask: 3D boolean array (slices, height, columns)
    """
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    nans = np.isnan(upper) | np.isnan(lower)
    # Avoid casting issues with NaNs
    upper = np.where(nans, -1, upper)
    lower = np.where(nans, -1, lower)
    # Round to nearest integer and clip to valid range
    upper_idx = np.rint(upper).astype(int)
    lower_idx = np.rint(lower).astype(int)
    upper_idx = np.clip(upper_idx, 0, height)
    lower_idx = np.clip(lower_idx, 0, height)

    valid = ~nans & (upper_idx < lower_idx)
    mask = np.zeros((upper.shape[0], height, upper.shape[1]), dtype=bool)
    for sli, col in zip(*np.where(valid)):
        mask[sli, upper_idx[sli, col]:lower_idx[sli, col], col] = True
    return mask
