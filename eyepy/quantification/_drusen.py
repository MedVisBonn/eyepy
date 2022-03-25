# -*- coding: utf-8 -*-
import logging

import numpy as np

from eyepy.core import EyeVolumeLayerAnnotation
from eyepy.quantification.utils.filter import filter_by_height_enface

logger = logging.getLogger("eyepy.quantification.drusen")


def ideal_rpe(rpe_height, bm_height, volume_shape):
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
    hist, edges = np.histogram(clean_shifted.flatten(), bins=np.arange(volume_shape[1]))

    # Compute the ideal RPE as the mean of the biggest bin and its neighbours
    lower_edge = edges[np.argmax(hist) - 1]
    upper_edge = edges[np.argmax(hist) + 2]
    irpe_height = np.mean(
        clean_shifted[
            np.logical_and(clean_shifted <= upper_edge, clean_shifted >= lower_edge)
        ]
    )
    ideal_rpe = np.full_like(shifted_rpe_height, irpe_height)

    # Shift back into original image space
    ideal_rpe = np.reshape(ideal_rpe, (d, w)) - shift

    return ideal_rpe


def drusen(rpe_height, bm_height, volume_shape, minimum_height=2):
    """Compute drusen from the RPE and BM layer segmentation.

    First estimate the ideal RPE based on a histogram of the RPE heights relativ
    to the BM.
    Then compute drusen as the area between the RPE and the normal RPE
    """
    # Estimate ideal RPE
    if type(rpe_height) is EyeVolumeLayerAnnotation:
        rpe_height = rpe_height.data
    if type(bm_height) is EyeVolumeLayerAnnotation:
        bm_height = bm_height.data

    idealrpe = ideal_rpe(rpe_height, bm_height, volume_shape)
    # Create drusen map
    drusen_map = np.zeros(volume_shape, dtype=bool)
    # Exclude normal RPE and RPE from the drusen area.
    rpe = np.flip(np.rint(rpe_height + 1).astype(int), axis=0)
    irpe = np.flip(np.rint(idealrpe).astype(int), axis=0)
    for sli in range(drusen_map.shape[0]):
        for col in range(drusen_map.shape[2]):
            if not rpe[sli, col] == -9223372036854775808:
                drusen_map[sli, rpe[sli, col] : irpe[sli, col], col] = 1

    drusen_map = filter_by_height_enface(drusen_map, minimum_height)

    return drusen_map
