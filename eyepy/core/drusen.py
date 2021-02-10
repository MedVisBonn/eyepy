from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d


class DrusenFinder(ABC):
    @abstractmethod
    def find(self, oct_obj):
        """ A function which returns a boolean map indicating drusen"""
        raise NotImplementedError()

    @abstractmethod
    def filter(self, drusen_map):
        """ A function which filters drusen based on the drusen properties """
        raise NotImplementedError()


class DefaultDrusenFinder(DrusenFinder):
    def __init__(self, degree=3, iterations=5, outlier_threshold=5,
                 poly_fit_type="regularized", minimum_height=2, minimum_depth=2,
                 minimum_area=0, minimum_volume=0, voxel_size=(1, 1, 1)):
        """

        Parameters
        ----------
        degree: Maximum degree of the Polynomial used for finding the normal RPE
        iterations: Number of iterations in the outlier removal
        outlier_threshold: Threshold for RPE Pixel to be considered in the next iteration
        poly_fit_type: If "regularized", regularize the normal RPE fitting.
        minimum_height: Minimum drusen  height.
        minimum_area: Minimum enface area. Currently not used
        minimum_volume: Minimum drusen volume. Currently not used
        voxel_size: (height, width, depth) length of the voxel in µm.
        """
        self.degree = degree
        self.iterations = iterations
        self.outlier_threshold = outlier_threshold
        self.poly_fit_type = poly_fit_type
        self.minimum_height = minimum_height
        self.minimum_volume = minimum_volume
        self.minimum_area = minimum_area
        self.minimum_depth = minimum_depth
        self.voxel_size = voxel_size

    def filter(self, drusen_map):
        return drusen_map
        #return filter_by_height(drusen_map, self.minimum_height,
        #                        self.voxel_size)
        #return filter_by_depth(d, self.minimum_depth)

    def find(self, oct_obj):
        drusen_map = np.zeros(oct_obj.shape, dtype=bool)
        for i, scan in enumerate(oct_obj):
            try:
                drusen_map[..., i] = drusen(
                    scan.layers["RPE"], scan.layers["BM"], scan.shape,
                    degree=self.degree, iterations=self.iterations,
                    outlier_threshold=self.outlier_threshold,
                    poly_fit_type=self.poly_fit_type)
            except KeyError as e:
                print(e)

        return drusen_map


def drusen(rpe_height, bm_height, scan_shape, degree=3, iterations=5,
           outlier_threshold=5, poly_fit_type="regularized"):
    """ Compute drusen from the RPE layer segmentation.

    First estimate the normal RPE by fitting a polynomial to the RPE. Then
    compute drusen as the area between the RPE and the normal RPE
    """

    rpe_height = np.rint(interpolate_layer(rpe_height))
    bm_height = np.rint(interpolate_layer(bm_height))
    # Estimate normal RPE
    normal_rpe_height = normal_rpe(rpe_height, bm_height, scan_shape, degree,
                                   iterations, outlier_threshold,
                                   poly_fit_type)
    # Create drusen map
    drusen_map = np.zeros(scan_shape, dtype=bool)

    # Exclude normal RPE and RPE from the drusen area.
    for col in range(drusen_map.shape[1]):
        drusen_map[(rpe_height + 1).astype(int)[col]: normal_rpe_height.astype(int)[col], col] = 1
    return drusen_map

def interpolate_layer(layer_height, kind="nearest"):
    nans = np.isnan(layer_height)
    x = np.arange(layer_height.shape[0])[~nans]
    y = layer_height[~nans]
    f = interp1d(x, y, kind=kind, fill_value="extrapolate")
    interpolated_layer = np.copy(layer_height)
    interpolated_layer[nans] = f(np.argwhere(nans)[:, 0])
    return interpolated_layer


def normal_rpe(rpe_height, bm_height, scan_shape, degree=3, iterations=5,
               outlier_threshold=5, poly_fit_type="regularized"):
    """ Estimate the normal RPE
    First the shift to make the BM a horizontal line is found. Then this shift
    is applied to the RPE. A third degree polynomial is fitted to the shifted
    RPE and the resulting normal RPE curve is shifted back into the original
    image space.
    """
    h, w = scan_shape

    # compute shift needed to align the BM to the horizontal center line
    shift = np.empty(w, dtype='int')
    shift.fill(h - (h / 2))
    shift = shift - bm_height

    # now shift the RPE location vector as well
    shifted_rpe_height = rpe_height + shift

    # These variables change while outliers are removed and polynomial fitting
    tmpx = range(w)
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
            tmpx = np.argwhere(inlier)[:, 0]
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


def filter_by_depth(drusen_map, minimum_depth=2):
    filtered_drusen = np.copy(drusen_map)
    if minimum_depth == 0:
        return drusen_map
    # get array where connected components get same label
    connected_component_array, num_drusen = ndimage.label(drusen_map)
    # Go through each component, sum it along 2 axis and check max depth against threshold
    max_depths = np.zeros_like(connected_component_array)
    for label, drusen_pos in enumerate(ndimage.find_objects(connected_component_array)):
        component_sub_vol = connected_component_array[drusen_pos]
        component_max_depth = np.max(np.sum(component_sub_vol==label+1, axis=2))
        component_sub_vol[component_sub_vol==label+1] = component_max_depth
        max_depths[drusen_pos] = component_sub_vol
    filtered_drusen[max_depths < minimum_depth] = False
    return filtered_drusen.astype(bool)

def filter_by_height(drusen_map, minimum_height=2, voxel_size=(1, 1, 1)):
    if minimum_height == 0:
        return drusen_map
    connected_component_array, num_drusen = ndimage.label(drusen_map)
    component_height_array = component_max_height(connected_component_array)

    filtered_drusen = np.copy(drusen_map)
    filtered_drusen[component_height_array < minimum_height] = False
    return filtered_drusen.astype(bool)

def component_max_height(connected_component_array):
    #labels = np.unique(connected_component_array)
    max_heights = np.zeros_like(connected_component_array)
    for label, drusen_pos in enumerate(ndimage.find_objects(connected_component_array)):
        component_sub_vol = connected_component_array[drusen_pos]
        component_max_height = np.max(np.sum(component_sub_vol==label, axis=0))
        component_sub_vol[component_sub_vol==label] = component_max_height
        max_heights[drusen_pos] = component_sub_vol
    return max_heights

def filter_by_width(drusen_map, minimum_width=2):
    """ Filter drusen by width in single B-Scans"""
    filtered_drusen = np.copy(drusen_map)
    if minimum_width == 0:
        return drusen_map
    
    for i in range(filtered_drusen.shape[-1]):
        scan = filtered_drusen[..., i]
        drusen_cols = np.any(filtered_drusen, axis=0)
        connected_drusen, n = ndimage.label(drusen_cols)
        if n == 0:
            continue
        for drusen_pos in ndimage.find_objects(connected_drusen):
            if np.sum(drusen_cols[drusen_pos]) < minimum_width:
                filtered_drusen[:, drusen_pos[0], i] = False
    return filtered_drusen.astype(bool)


def filter_by_area():
    pass

def filter_by_volume():
    pass
