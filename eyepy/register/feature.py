# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import hog


def hog_extract(
    img,
    orientations=12,
    pixels_per_cell=(5, 5),
    cells_per_block=(20, 20),
    rotate_hist=False,
    sub_sample_factor=2,
):
    """"""

    hog_features = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        multichannel=False,
        feature_vector=False,
    )

    ss_hog_features = hog_features[::sub_sample_factor, ::sub_sample_factor, ...]

    if rotate_hist:
        ss_hog_features = rotate_hog(ss_hog_features)

    # flatten blocks
    ss_hog_features = ss_hog_features.reshape(ss_hog_features.shape[:-3] + (-1,))
    keys = []
    features = []
    for key in np.ndindex(ss_hog_features.shape[:-1]):
        feature = ss_hog_features[key]
        keys.append(key)
        features.append(feature)

    features = np.array(features)
    keys = np.array(keys)

    # Rescale features to original image positions
    N0, N1 = pixels_per_cell
    M0, M1 = cells_per_block
    keys[:, 0] = keys[:, 0] * sub_sample_factor * N0 + (N0 * M0 / 2)
    keys[:, 1] = keys[:, 1] * sub_sample_factor * N1 + (N1 * M1 / 2)

    return keys, features


def rotate_hog(hog_features):
    """"""
    orientations = hog_features.shape[-1]
    M = hog_features.shape[-2]
    rotated_hog = np.zeros(hog_features.shape[:2] + (orientations, M, M, orientations))
    for block in np.ndindex(hog_features.shape[:2]):
        for rot in range(orientations):
            rotated_hog[block][rot] = np.roll(hog_features[block], rot, axis=-1)

    return rotated_hog
