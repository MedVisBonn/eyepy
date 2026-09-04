from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def compute_retina_mask(
    image: npt.NDArray[np.float64],
    threshold: int = 2,
    max_iterations: int = 10,
    upper: npt.NDArray[np.int_] | None = None,
    lower: npt.NDArray[np.int_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Create a retina mask for the input image.

    Args:
        image: Input image array.
        threshold: Threshold for removing outliers.
        max_iterations: Maximum number of iterations for outlier removal.
        upper: Upper boundary of the retina.
        lower: Lower boundary of the retina.

    Returns:
        A binary mask for the input image.

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError(
            "The 'scipy' package is required for retina detection. "
            'Please install it with `pip install scipy` or `pip install eyepy[quant]`.'
        )

    # Remove top and bottom 5% of the image to remove noisy border intensities
    logger.debug('Computing retina mask')
    image = np.copy(image)
    image_height = image.shape[0]
    border_cut = int(image_height * 0.05)
    image[:border_cut, :] = 0
    image[-border_cut:, :] = 0
    logger.debug(f'Gaussian filtering: {image.dtype}')

    image[np.isnan(image)] = np.nanmean(image)
    image = ndimage.gaussian_filter(image, 3)

    logger.debug('Peak finder')
    result = np.apply_along_axis(peak_finder, 0, image)
    logger.debug('Peak finder done')

    # replace nan values with closest non-nan value
    # result shape is (2, width)
    # Check if there are valid peaks
    valid_upper = result[0] != -1
    valid_lower = result[1] != -1

    if np.any(valid_upper):
        result[0, ~valid_upper] = np.mean(result[0][valid_upper])
    else:
        # Fallback if no peaks found
        result[0, :] = image_height // 3

    if np.any(valid_lower):
        result[1, ~valid_lower] = np.mean(result[1][valid_lower])
    else:
        # Fallback if no peaks found
        result[1, :] = 2 * image_height // 3

    logger.debug('Fitting polynomials')
    x_axis = np.arange(image.shape[1])

    if upper is None:
        coeffs, _, _ = iteratively_remove_outliers(
            result[0], threshold=threshold, max_iterations=max_iterations
        )
        upper = np.poly1d(coeffs)(x_axis) - 10

    if lower is None:
        coeffs, _, _ = iteratively_remove_outliers(
            result[1], threshold=threshold, max_iterations=max_iterations
        )
        lower = np.poly1d(coeffs)(x_axis) + 10

    upper = np.rint(upper).astype(int)
    lower = np.rint(lower).astype(int)
    mask = np.zeros_like(image, dtype=bool)

    # Clip boundaries to image dimensions
    upper = np.clip(upper, 0, image_height)
    lower = np.clip(lower, 0, image_height)

    for col in range(mask.shape[1]):
        mask[upper[col]:lower[col], col] = True

    return mask


def peak_finder(data: npt.NDArray[np.float64],
                window_size: int = 10) -> npt.NDArray[np.int_]:
    """Find the two highest prominences in the given data.

    Args:
        data: Input data array (1D).
        window_size: Window size for convolution operation.

    Returns:
        An array containing the indices of the two highest prominences.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError(
            "The 'scipy' package is required for retina detection. "
            'Please install it with `pip install scipy` or `pip install eyepy[quant]`.'
        )

    # Convolve the data with a window for smoothing
    kernel = np.ones(window_size) / window_size
    data = np.convolve(data, kernel, mode='same')
    data = np.convolve(data, kernel, mode='same')

    peaks, properties = find_peaks(data, prominence=1 / 250)
    prominences = properties['prominences']

    if len(peaks) < 2:
        return np.array([-1, -1])

    highest_prominence_indices = np.argsort(prominences)[-2:]
    return peaks[np.sort(highest_prominence_indices)]


def remove_outliers(
    data: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    threshold: int = 2
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Remove outlier data points based on the given threshold.

    Args:
        data: Input data array.
        residuals: Residuals of the data.
        threshold: Threshold for removing outliers.

    Returns:
        A tuple containing the new data and mask after removing outliers.
    """
    mean = np.mean(residuals)
    std_dev = np.std(residuals)
    mask = np.abs(residuals - mean) < threshold * std_dev
    return data[mask], mask


def iteratively_remove_outliers(
    data: npt.NDArray[np.float64],
    x: npt.NDArray[np.int_] | None = None,
    threshold: int = 3,
    max_iterations: int = 5,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.int_]]:
    """Iteratively remove outliers from the data and fit a polynomial.

    Args:
        data: Input data array.
        x: X-axis values. Defaults to None.
        threshold: Threshold for removing outliers.
        max_iterations: Maximum number of iterations for outlier removal.

    Returns:
        A tuple containing the polynomial coefficients, new data, and x values.
    """
    if x is None:
        x = np.arange(len(data))

    coeffs = np.polyfit(x, data, 3)

    for _ in range(max_iterations):
        logger.debug(f'Fit iteration, data length {len(data)}')
        coeffs = np.polyfit(x, data, 3)
        poly = np.poly1d(coeffs)
        residuals = data - poly(x)
        new_data, mask = remove_outliers(data, residuals, threshold)
        new_x = x[mask]

        if len(new_data) == len(data):
            break

        data = new_data
        x = new_x

    return coeffs, data, x
