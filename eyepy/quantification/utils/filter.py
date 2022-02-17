# -*- coding: utf-8 -*-
import logging

import numpy as np
from scipy import ndimage as ndimage

logger = logging.getLogger(__name__)


def filter_by_depth(drusen_map, minimum_depth=2):
    filtered_drusen = np.copy(drusen_map)
    if minimum_depth == 0:
        return drusen_map
    # get array where connected components get same label
    connected_component_array, num_drusen = ndimage.label(drusen_map)
    # Go through each component, sum it along axis 0 and check max depth against threshold
    max_depths = np.zeros_like(connected_component_array)
    for label, drusen_pos in enumerate(ndimage.find_objects(connected_component_array)):
        component_sub_vol = connected_component_array[drusen_pos]
        component_max_depth = np.max(np.sum(component_sub_vol == label + 1, axis=0))
        component_sub_vol[component_sub_vol == label + 1] = component_max_depth
        max_depths[drusen_pos] = component_sub_vol
    filtered_drusen[max_depths < minimum_depth] = False
    return filtered_drusen.astype(bool)


def filter_by_height_enface(drusen_map, minimum_height=2):
    if minimum_height == 0:
        return drusen_map

    projection = np.sum(drusen_map, axis=1, keepdims=True)  # Shape (n_bscans, width)

    # Find connected components in the enface projection
    connected_component_array, num_drusen = ndimage.label(projection != 0)
    # print(connected_component_array.shape)
    max_heights = np.zeros_like(connected_component_array)
    for drusen_pos in ndimage.find_objects(connected_component_array):
        # Work on subvolume for faster processing
        component_sub_vol = connected_component_array[drusen_pos]
        # Find current label (most frequent label in the subvolume)
        label = np.bincount(component_sub_vol[component_sub_vol != 0]).argmax()
        component_max_height = np.max(
            projection[drusen_pos][component_sub_vol == label]
        )
        # Set drusen region to drusen max height
        max_heights[drusen_pos][component_sub_vol == label] = component_max_height

    filtered_drusen = np.copy(drusen_map)
    indices = np.nonzero(max_heights < minimum_height)
    filtered_drusen[indices[0], :, indices[2]] = False
    return filtered_drusen.astype(bool)
