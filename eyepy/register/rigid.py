# -*- coding: utf-8 -*-
from typing import Tuple, Union, Callable, Mapping, Any

import eyepy as ep
import numpy as np
import skimage as ski

import matplotlib.pyplot as plt


Shape = Union[int, Tuple[int, int]]
Detector = Union[str, Tuple[np.ndarray, np.ndarray], Callable[..., np.ndarray]]
Descriptor = Union[str, Callable[..., np.ndarray]]
Matching = Union[str, Callable[..., np.ndarray]]
Transformation = str


def _get_func(arg, mapping):
    if callable(arg):
        return arg
    elif isinstance(arg, str):
        return mapping[arg]
    else:
        raise ValueError(
            "{} is not allowed. Provide a callable or one of {}".format(
                arg, mapping.keys()
            )
        )


def register(
    src_img: np.ndarray,
    dest_img: np.ndarray,
    src_roi: np.ndarray = None,
    dest_roi: np.ndarray = None,
    scale: int = 1,
    preprocessing: Union[str, Callable[..., np.ndarray]] = None,
    detector: Detector = "dense",
    descriptor: Descriptor = "mind",
    matching: Matching = "ann",
    transformation: Transformation = "similarity",
    verbose: bool = False,
    preprocessing_kwargs: Mapping[str, Any] = None,
    detector_kwargs: Mapping[str, Any] = None,
    descriptor_kwargs: Mapping[str, Any] = None,
    matching_kwargs: Mapping[str, Any] = None,
    ransac_kwargs: Mapping[str, Any] = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    src_img :
    dest_img :
    src_roi :
    dest_roi :
    scale :
    preprocessing : The identifier of the method or a function used for preprocessing. Available options: ("meanphase").
    You can provide a mapping of keyword arguments for this step using the "preprocessing_kwargs" argument.
    detector :
    descriptor :
    matching : The identifier of the matching method. Available options ("ann", "nn"). A metric and other keyword
    arguments can be specified in the matching_kwargs argument. You can also provide your own matching function here.
    transformation :The identifier of the transformation model. Internally we are using skimage here.
    Available options: ("euclidean", "similarity", "affine", "projective")
    verbose : Whether or not to plot intermediate results
    preprocessing_kwargs : Keyword arguments for the preprocessing step
    detector_kwargs : Keyword arguments for the keypoint detection step
    descriptor_kwargs : Keyword arguments for the keypoint description step
    matching_kwargs : Keyword arguments for the keypoint matching step
    ransac_kwargs: Keyword arguments for the  RANSAC process used for finding a robust transformation

    Returns
    -------

    """
    if preprocessing_kwargs is None:
        preprocessing_kwargs = {}
    if detector_kwargs is None:
        detector_kwargs = {}
    if descriptor_kwargs is None:
        descriptor_kwargs = {}
    if matching_kwargs is None:
        matching_kwargs = {}
    if ransac_kwargs is None:
        ransac_kwargs = {"residual_threshold": 5}

    # Preprocessing
    PREPROCESSING = {"meanphase": ep.preprocess.mean_phase}

    if preprocessing:
        preprocessing_func = _get_func(preprocessing, PREPROCESSING)
        src_p = preprocessing_func(src_img, **preprocessing_kwargs)
        dest_p = preprocessing_func(dest_img, **preprocessing_kwargs)
    else:
        src_p = src_img
        dest_p = dest_img

    # Resize images
    src_p = ski.transform.rescale(src_p, scale, anti_aliasing=True)
    dest_p = ski.transform.rescale(dest_p, scale, anti_aliasing=True)

    # Keypoint detection
    # TODO: Find keypoints and include them in the later workflow. -> Sparse features/ROIs
    DETECTOR = {"CENSURE": ski.feature.CENSURE, "FAST": ski.feature.corner_fast}
    if detector == "dense":
        keypoints = None
    else:
        raise NotImplementedError("Currently only dense features are supported")
        detector = _get_func(detector, DETECTOR)()
        detector.detect(src_p)
        src_keypoints = detector.keypoints

        detector.detect(dest_p)
        dest_keypoints = detector.keypoints

    # Keypoint description
    DESCRIPTOR = {
        "mind": ep.register.feature.mind_extract,
        "hog": ep.register.feature.hog_extract,
        "daisy": ep.register.feature.daisy_extract,
    }

    descriptor_func = _get_func(descriptor, DESCRIPTOR)

    # TODO: Hand keypoints to the descriptor -> connect sparse descriptors
    dest_keys, dest_features = descriptor_func(dest_p, **descriptor_kwargs)
    src_keys, src_features = descriptor_func(src_p, **descriptor_kwargs)

    # Apply ROI
    if src_roi is not None:
        # clean src keys
        src_roi_small = ski.transform.rescale(src_roi, scale).astype(bool)
        src_selection = src_roi_small[
            src_keys[:, 0].astype(int), src_keys[:, 1].astype(int)
        ]
        src_keys = src_keys[src_selection]
        src_features = src_features[src_selection]

    if dest_roi is not None:
        # clean dest keys
        dest_roi_small = ski.transform.rescale(dest_roi, scale).astype(bool)
        dest_selection = dest_roi_small[
            dest_keys[:, 0].astype(int), dest_keys[:, 1].astype(int)
        ]
        dest_keys = dest_keys[dest_selection]
        dest_features = dest_features[dest_selection]

    # Keypoint matching
    MATCHING = {
        "nn": ski.feature.match_descriptors,
        "ann": ep.register.feature.match_ann_descriptors,
    }
    matching_func = _get_func(matching, MATCHING)
    matches = matching_func(
        dest_features, src_features, cross_check=True, **matching_kwargs
    )

    # Estimate transformation

    # Switch row, col to col, row order for the keys and compensate the
    # downscaling to produce an affine for the original image
    dest_keys = (dest_keys / scale)[:, [1, 0]]
    src_keys = (src_keys[:, :2] / scale)[:, [1, 0]]

    TRANSFORMS = {
        "euclidean": [ski.transform.EuclideanTransform, 2],
        "similarity": [ski.transform.SimilarityTransform, 3],
        "affine": [ski.transform.AffineTransform, 3],
        "projective": [ski.transform.ProjectiveTransform, 4],
    }
    model_robust, inliers = ski.measure.ransac(
        (src_keys[matches[:, 1]], dest_keys[matches[:, 0]]),
        TRANSFORMS[transformation][0],
        min_samples=TRANSFORMS[transformation][1],
        **ransac_kwargs
    )

    if verbose:
        dest_keys = dest_keys[:, [1, 0]]
        src_keys = src_keys[:, [1, 0]]

        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        ski.feature.plot_matches(
            ax,
            dest_img,
            src_img,
            dest_keys,
            src_keys,
            matches[inliers],
            keypoints_color="green",
            only_matches=True,
        )

    return model_robust
