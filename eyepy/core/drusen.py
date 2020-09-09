import numpy as np
from scipy.interpolate import interp1d


def drusen(rpe_height, bm_height, scan_shape, degree=3, iterations=5,
           outlier_threshold=5, poly_fit_type="regularized"):
    """ Compute drusen from the RPE layer segmentation.

    First estimate the normal RPE by fitting a polynomial to the RPE. Then
    compute drusen as the area between the RPE and the normal RPE
    """
    # Estimate normal RPE
    normal_rpe_height = normal_rpe(rpe_height, bm_height, scan_shape, degree,
                                   iterations, outlier_threshold,
                                   poly_fit_type)
    # Create drusen map
    drusen_map = np.zeros(scan_shape)
    # Exclude normal RPE and RPE from the drusen area.
    drusen_map[normal_rpe_height + 1:rpe_height] = 1
    return drusen_map


def normal_rpe(rpe_height, bm_height, scan_shape, degree=3, iterations=5,
               outlier_threshold=5, poly_fit_type="regularized"):
    """ Estimate the normal RPE

    First the shift to make the BM a horizontal line is found. Then this shift
    is applied to the RPE. A third degree polynomial is fitted to the shifted
    RPE and the resulting normal RPE curve is shifted back into the original
    image space.
    """
    h, w = scan_shape

    # interpolate NANs in BM
    nans = np.isnan(bm_height)
    f = interp1d(np.arange(w)[~nans], bm_height[~nans], kind="nearest")
    bm_height[nans] = f(np.argwhere(nans))

    # compute shift needed to align the BM to the horizontal center line
    shift = np.empty(w, dtype='int')
    shift.fill(h - (h / 2))
    shift = shift - bm_height

    # now shift the RPE location vector as well
    shifted_rpe_height = rpe_height + shift

    # These variables change while outliers are removed and polynomial fitting
    tmpx = range(len(rpe_height))
    tmpy = np.copy(shifted_rpe_height)

    it = 0
    while True:
        if poly_fit_type == 'regularized':
            coeffs = compute_regularized_fit(tmpx, tmpy, deg=degree)
        else:
            coeffs = np.polyfit(tmpx, tmpy, deg=degree)

        # Evaluate the polynomial for all x values
        norm_rpe = np.polyval(coeffs, range(w)).astype('int')

        # Outlier removal
        if outlier_threshold:
            # Compute norm rpe with these values in next iteration
            inlier = norm_rpe - shifted_rpe_height < outlier_threshold
            tmpx = np.argwhere(inlier)
            tmpy = shifted_rpe_height[inlier]
        else:
            # TODO: Is this correct
            # Element wise maximum of the polynomial fitted rpe and the rpe
            tmpy = np.maximum(norm_rpe, tmpy)

        it += 1
        if it >= iterations:
            break

    # Shift back into original image space
    norm_rpe = norm_rpe - shift

    return norm_rpe


def compute_regularized_fit(x, y, deg):
    result_matrix = np.zeros((deg + 1, deg + 1))
    for d in range(deg + 1):
        z = np.polyfit(x, y, deg=d)
        for i in range(len(z)):
            result_matrix[d, -1 - i] = z[-1 - i]
    # The highest degree has the lowest weight
    weighted_average = np.average(result_matrix, axis=0,
                                  weights=[1., 1., 0.1 * 2, 0.1 ** 4])
    return weighted_average
