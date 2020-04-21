# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import Tuple, Union

import numpy as np

Shape = Union[int, Tuple[int, int]]



# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import Tuple, Union

import numpy as np

Shape = Union[int, Tuple[int, int]]


@lru_cache(maxsize=8)
def filtergrid(size: Shape, quadrant_shift: bool =True, normalize:bool =True) -> Tuple[np.ndarray, np.ndarray]:
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
def radius_filtergrid(size: Shape, quadrant_shift: bool =True, normalize:bool =True) -> np.ndarray:
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
def theta_filtergrid(size: Shape, quadrant_shift: bool =True) -> np.ndarray:
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


@lru_cache(maxsize=8)
def lowpassfilter(size: Shape, cutoff: float, order: int) -> np.ndarray:
    """Constructs a low-pass butterworth filter.

    Parameters
    ----------
    size : Size of the filter
    cutoff : Cutoff frequency of the filter 0 - 0.5
    order : Order of the filter, the higher n is the sharper the transition is.
    (n must be an integer >= 1). Note that n is doubled so that it is always an
    even integer.

    Returns
    -------

    The filter is compute using the following formula:
    filter = 1.0 / (1.0 + (radius / cutoff) ^ (2 * order))

    The frequency origin of the returned filter is at the corners.

    Inspired by lowpassfilter.m found at https://www.peterkovesi.com/matlabfns/
    """
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("The cutoff frequency must be between 0 and 0.5")

    if order != int(order) or order < 1:
        raise ValueError("order must be an iteger >= 1")

    # Construct spatial frequency values in terms of normalised radius from centre.
    radius = radius_filtergrid(size)

    lp_filter = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))

    return lp_filter


@lru_cache(maxsize=8)
def _log_gabor_radial(size: Shape, wavelength: float, sigma: float) -> np.ndarray:
    """

    Parameters
    ----------
    size :
    wavelength :
    sigma :

    Returns
    -------

    """
    freq = 1.0 / wavelength
    radius = radius_filtergrid(size)
    radius[0, 0] = 1e-10
    radial_spread = np.exp((-((np.log(radius / freq)) ** 2)) / (2 * np.log(sigma) ** 2))

    # Construct low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries. The radial part of the log Gabor
    # filter is multiplied by the low-pass filter to ensure no extra frequencies
    # at the 'corners' of the FFT are incorporated.
    lp = lowpassfilter(size, 0.45, 15)
    # Radius .45, 'sharpness' 15
    # Apply low-pass filter
    radial_spread = radial_spread * lp
    # Set the value at the 0 frequency point of the filter
    # back to zero (undo the radius fudge).
    radial_spread[0, 0] = 0

    return radial_spread


@lru_cache(maxsize=8)
def _log_gabor_angular(size: Shape, angle: float, angular_frac: float) -> np.ndarray:
    # For each point in the filter matrix calculate the angular distance from
    # the specified filter orientation.  To overcome the angular wrap-around
    # problem sine difference and cosine difference values are first computed
    # and then the atan2 function is used to determine angular distance.
    theta = theta_filtergrid(size)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
    # Difference in sine.
    dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
    # Difference in cosine.
    dtheta = np.abs(np.arctan2(ds, dc))
    # Absolute angular distance.
    # Scale theta so that cosine spread function has the right wavelength and clamp to pi
    dtheta = np.minimum(dtheta / angular_frac / 2, np.pi)
    # The spread function is cos(dtheta) between -pi and pi.  We add 1,
    # and then divide by 2 so that the value ranges 0-1
    angular_spread = (np.cos(dtheta) + 1) / 2

    return angular_spread


def log_gabor_kernel(
    size: Shape,
    wavelength: float = 3,
    sigma: float = 0.55,
    angle: float = 0.0,
    angular_frac: float = 1 / 6,
) -> np.ndarray:
    """

    Parameters
    ----------
    size :
    wavelength :
    sigma :
    angle :
    angular_frac :

    Returns
    -------

    """

    # Radial component which controls the frequency band that the filter responds to
    radial_spread = _log_gabor_radial(size, wavelength, sigma)
    # The angular component, which controls the orientation that the filter responds to.
    angular_spread = _log_gabor_angular(size, angle, angular_frac)
    log_gabor_filter = radial_spread * angular_spread
    return log_gabor_filter


def log_gabor(
    image: np.ndarray,
    wavelength: float = 3,
    sigma: float = 0.55,
    angle: float = 0.0,
    angular_frac: float = 1 / 6,
) -> np.ndarray:
    """

    Parameters
    ----------
    image :
    wavelength :
    sigma :
    angle :
    angular_frac :

    Returns
    -------

    """
    image_fft = np.fft.fft2(image)

    kernel = log_gabor_kernel(image.shape, wavelength, sigma, angle, angular_frac)

    even_odd = np.fft.ifft2(image_fft * kernel)

    return even_odd


def mean_phase(
    image: np.ndarray,
    min_wavelength: float = 3,
    sigma: float = 0.55,
    n_scale: int = 4,
    mult: float = 2.1,
    n_orient: int = 6,
) -> np.ndarray:
    """

    Parameters
    ----------
    image :
    min_wavelength :
    sigma :
    n_scale :
    mult :
    n_orient :

    Returns
    -------

    """
    phase_sum = np.zeros(image.shape)
    count = 0
    for scale in range(n_scale):
        wavelength = min_wavelength * mult ** scale
        for orient in range(n_orient):
            angle = orient * np.pi / n_orient
            eo = log_gabor(
                image, wavelength, sigma, angle=angle, angular_frac=1 / n_orient
            )
            phase = np.arctan2(np.abs(eo.imag), eo.real)
            ampl = np.sqrt(eo.imag ** 2 + eo.real ** 2)
            phase_sum += phase * ampl
            count += ampl

    return phase_sum / count


def phase_congruency(
    image: np.ndarray,
    min_wavelength: float = 3,
    sigma: float = 0.55,
    n_scale: int = 4,
    mult: float = 2.1,
    n_orient: int = 6,
    noise_method: int = -1,
    k: float = 2.0,
    cutoff: float = 0.5,
    g: int = 10,
):

    """

    Parameters
    ----------
    image :
    min_wavelength :
    sigma :
    n_scale :
    mult :
    n_orient :
    noise_method:
    k: No of standard deviations of the noise energy beyond the mean at which we set the noise threshold point.
    You may want to vary this up to a value of 10 or20 for noisy images
    cutoff: The fractional measure of frequency spread below which phase congruency values get penalized.
    g: Controls the sharpness of the transition in the sigmoid function used to weight phase congruency for
    frequency spread.

    Returns
    -------

    """
    epsilon = 0.00001
    pcSum = np.zeros(image.shape)
    covx2 = np.zeros(image.shape)
    covy2 = np.zeros(image.shape)
    covxy = np.zeros(image.shape)
    eo_filter_responses = {}
    EnergyV = np.zeros(image.shape + (3,))
    pc = {}
    for orient in range(n_orient):
        angle = orient * np.pi / n_orient
        sumE_ThisOrient = np.zeros(image.shape)  # Initialize accumulator matrices.
        sumO_ThisOrient = np.zeros(image.shape)
        sumA_ThisOrient = np.zeros(image.shape)
        Energy = np.zeros(image.shape)
        for scale in range(n_scale):
            wavelength = min_wavelength * mult ** scale

            eo = log_gabor(
                image, wavelength, sigma, angle=angle, angular_frac=1 / n_orient
            )
            amplitude = abs(eo)
            sumA_ThisOrient = sumA_ThisOrient + amplitude
            sumE_ThisOrient = sumE_ThisOrient + eo.real
            sumO_ThisOrient = sumO_ThisOrient + eo.imag

            eo_filter_responses[(orient, scale)] = eo

            # At the smallest scale estimate noise characteristics from the
            # distribution of the filter amplitude responses stored in sumAn.
            # tau is the Rayleigh parameter that is used to describe the
            # distribution.
            if scale == 0:
                if noise_method == -1:  # Use median to estimate noise statistics
                    tau = np.median(sumA_ThisOrient / np.sqrt(np.log(4)))
                # elif noise_method == -2:   # Use mode to estimate noise statistics
                #    tau = rayleighmode(sumAn_ThisOrient(:))
                maxA = amplitude
            else:
                # Record maximum amplitude of components across scales.  This is needed
                # to determine the frequency spread weighting.
                maxA = np.maximum(maxA, amplitude)

        # Accumulate total 3D energy vector data, this will be used to
        # determine overall feature orientation and feature phase/type
        EnergyV[:, :, 0] = EnergyV[:, :, 0] + sumE_ThisOrient
        EnergyV[:, :, 1] = EnergyV[:, :, 1] + np.cos(angle) * sumO_ThisOrient
        EnergyV[:, :, 2] = EnergyV[:, :, 2] + np.sin(angle) * sumO_ThisOrient

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.

        for scale in range(n_scale):
            # Extract even and odd convolution results.
            even = eo.real
            odd = eo.imag

            Energy = (
                Energy + even * MeanE + odd * MeanO - abs(even * MeanO - odd * MeanE)
            )

        # Automatically determine noise threshold
        #
        # Assuming the noise is Gaussian the response of the filters to noise will
        # form Rayleigh distribution.  We use the filter responses at the smallest
        # scale as a guide to the underlying noise level because the smallest scale
        # filters spend most of their time responding to noise, and only
        # occasionally responding to features. Either the median, or the mode, of
        # the distribution of filter responses can be used as a robust statistic to
        # estimate the distribution mean and standard deviation as these are related
        # to the median or mode by fixed constants.  The response of the larger
        # scale filters to noise can then be estimated from the smallest scale
        # filter response according to their relative bandwidths.
        #
        # This code assumes that the expected reponse to noise on the phase congruency
        # calculation is simply the sum of the expected noise responses of each of
        # the filters.  This is a simplistic overestimate, however these two
        # quantities should be related by some constant that will depend on the
        # filter bank being used.  Appropriate tuning of the parameter 'k' will
        # allow you to produce the desired output.

        if noise_method >= 0:  # We are using a fixed noise threshold
            T = noise_method
            # use supplied noiseMethod value as the threshold
        else:
            # Estimate the effect of noise on the sum of the filter responses as
            # the sum of estimated individual responses (this is a simplistic
            # overestimate). As the estimated noise response at succesive scales
            # is scaled inversely proportional to bandwidth we have a simple
            # geometric sum.
            totalTau = tau * (1 - (1 / mult) ** n_scale) / (1 - (1 / mult))

            # Calculate mean and std dev from tau using fixed relationship
            # between these parameters and tau. See
            # http://mathworld.wolfram.com/RayleighDistribution.html
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)  # Expected mean and std
            EstNoiseEnergySigma = totalTau * np.sqrt(
                (4 - np.pi) / 2
            )  # values of noise energy

            T = EstNoiseEnergyMean + k * EstNoiseEnergySigma  # Noise threshold

        # Apply noise threshold,  this is effectively wavelet denoising via
        # soft thresholding.
        Energy = np.maximum(Energy - T, 0)

        # Form weighting that penalizes frequency distributions that are
        # particularly narrow.  Calculate fractional 'width' of the frequencies
        # present by taking the sum of the filter response amplitudes and dividing
        # by the maximum amplitude at each point on the image.   If
        # there is only one non-zero component width takes on a value of 0, if
        # all components are equal width is 1.
        width = (sumA_ThisOrient / (maxA + epsilon) - 1) / (n_scale - 1)

        # Now calculate the sigmoidal weighting function for this orientation.
        weight = 1.0 / (1 + np.exp((cutoff - width) * g))

        # Apply weighting to energy and then calculate phase congruency
        pc[orient] = (
            weight * Energy / sumA_ThisOrient
        )  # Phase congruency for this orientatio

        pcSum = pcSum + pc[orient]

        # Build up covariance data for every point
        covx = pc[orient] * np.cos(angle)
        covy = pc[orient] * np.sin(angle)
        covx2 = covx2 + covx ** 2
        covy2 = covy2 + covy ** 2
        covxy = covxy + covx * covy

    ## Edge and Corner calculations
    # The following is optimised code to calculate principal vector
    # of the phase congruency covariance data and to calculate
    # the minimumum and maximum moments - these correspond to
    # the singular values.

    # First normalise covariance values by the number of orientations/2
    covx2 = covx2 / (n_orient / 2)
    covy2 = covy2 / (n_orient / 2)
    covxy = 4 * covxy / n_orient  # This gives us 2*covxy/(n_orient/2)
    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    M = (covy2 + covx2 + denom) / 2  # Maximum moment
    m = (covy2 + covx2 - denom) / 2  # ... and minimum moment

    # Orientation and feature phase/type computation
    comp_orient = np.arctan2(EnergyV[:, :, 2], EnergyV[:, :, 1])
    comp_orient[comp_orient < 0] = (
        comp_orient[comp_orient < 0] + np.pi
    )  # Wrap angles -pi..0 to 0..pi
    comp_orient = np.round(
        comp_orient * 180 / np.pi
    )  # Orientation in degrees between 0 and 180

    OddV = np.sqrt(EnergyV[:, :, 1] ** 2 + EnergyV[:, :, 2] ** 2)
    feat_type = np.arctan2(
        EnergyV[:, :, 0], OddV
    )  # Feature phase  pi/2 <-> white line,
    # 0 <-> step, -pi/2 <-> black line

    return M, m, comp_orient, feat_type, pc, eo_filter_responses, T
