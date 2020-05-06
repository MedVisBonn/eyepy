# -*- coding: utf-8 -*-
from typing import Tuple, Union

import eyepy as ep
import numpy as np
import skimage as ski

Shape = Union[int, Tuple[int, int]]


def rigid_multimodal(
    src_img: np.ndarray,
    dest_img: np.ndarray,
    scale: int = 1,
    mp_kwargs: dict = {
        "min_wavelength": 3,
        "sigma": 0.55,
        "n_scale": 4,
        "mult": 2.1,
        "n_orient": 6,
    },
    hog_kwargs: dict = {
        "orientations": 12,
        "pixels_per_cell": (10, 10),
        "cells_per_block": (5, 5),
        "sub_sample_factor": 1,
    },
    ransac_kwargs: dict = {"residual_threshold": 5},
) -> np.ndarray:
    """

    Parameters
    ----------
    src_img
    dest_img
    work_size
    mp_kwargs
    hog_kwargs
    ransac_kwargs

    Returns
    -------

    """

    # Resize images
    src_small = ski.transform.rescale(src_img, scale, anti_aliasing=True)
    dest_small = ski.transform.rescale(dest_img, scale, anti_aliasing=True)

    # Calculate Mean Phase Image
    src_mp = ep.preprocess.mean_phase(src_small, **mp_kwargs)
    dest_mp = ep.preprocess.mean_phase(dest_small, **mp_kwargs)

    # Detect Features - HOG
    dest_keys, dest_features = ep.register.feature.hog_extract(dest_mp, **hog_kwargs)
    src_keys, src_features = ep.register.feature.hog_extract(
        src_mp, rotate_hist=True, **hog_kwargs
    )

    # Match Features - > ANN
    matches = ski.feature.match_descriptors(
        dest_features, src_features, cross_check=True
    )

    # Switch row, col to col, row order for the keys and compensate the
    # downscaling to produce an affine for the original image
    dest_keys = (dest_keys / scale)[:, [1, 0]]
    src_keys = (src_keys[:, :2] / scale)[:, [1, 0]]

    # Compute Affine
    model_robust, inliers = ski.measure.ransac(
        (src_keys[matches[:, 1]], dest_keys[matches[:, 0]]),
        ski.transform.AffineTransform,
        min_samples=3,
        **ransac_kwargs
    )

    # Warp Image
    transformed_src = ski.transform.warp(
        src_img, model_robust.inverse, output_shape=dest_img.shape
    )

    return transformed_src, model_robust
