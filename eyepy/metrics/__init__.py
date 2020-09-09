import numpy as np


def overlap_ratio_volume(ground_truth, prediction):
    """ Compute overlap ratio (OR) as described in [1]

    [1] Wintergerst, M.W.M., Gorgi Zadeh, S. et al.
        Replication and Refinement of an Algorithm for Automated Drusen Segmentation on Optical Coherence Tomography.
        Sci Rep 10, 7395 (2020).
        https://doi.org/10.1038/s41598-020-63924-6
    """
    union = np.sum(np.logical_or(ground_truth, prediction))
    intersection = np.sum(np.logical_and(ground_truth, prediction))
    # If the union is 0, neither ground_truth nor prediction contain anything -> OR becomes 1
    return intersection / union if union > 0 else 1


def absolute_area_difference_volume(ground_truth, prediction, out_format="percent", resolution=1):
    """ Compute the absolate area difference (AAD).

    This method is described in [1] Since it was used for drusen it was called ADAD

    [1] Wintergerst, M.W.M., Gorgi Zadeh, S. et al.
        Replication and Refinement of an Algorithm for Automated Drusen Segmentation on Optical Coherence Tomography.
        Sci Rep 10, 7395 (2020).
        https://doi.org/10.1038/s41598-020-63924-6
    """
    abs_area_diff = np.abs(np.sum(ground_truth) - np.sum(prediction))
    num_ascans = float(np.sum(np.sum(ground_truth, axis=0) > 0)) if np.sum(ground_truth) > 0.0 else 1.0
    abs_area_diff = float(abs_area_diff) / num_ascans
    gtLoad = float(np.sum((ground_truth > 0).astype('int'))) / num_ascans if num_ascans > 0 else 1.0
    if (out_format == "percent"):
        abs_area_diff = abs_area_diff / gtLoad if gtLoad > 0 else abs_area_diff
        abs_area_diff = abs_area_diff * 100
    elif (out_format == "um"):
        abs_area_diff = abs_area_diff * resolution
    else:
        pass
    return abs_area_diff


def absolute_area_difference(ground_truth, prediction, out_format="percent", resolution=1):
    absolute_area_difference_volume(ground_truth, prediction, out_format, resolution)


def overlap_ratio(ground_truth, prediction):
    overlap_ratio_volume(ground_truth, prediction)
