from skimage.feature import ORB
from skimage.feature import match_descriptors
from skimage.transform import AffineTransform, warp
import numpy as np
import eyepy as ep
from skimage.measure import ransac


def orb_detect_extract(img, **kwargs):
    descriptor_extractor = ORB(**kwargs)
    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def multimodal_2D(
    src_img: np.ndarray,
    dest_img: np.ndarray,
    mp_kwargs: dict = {},
    orb_kwargs: dict = {"n_keypoints": 200},
    ransac_kwargs: dict = {"residual_threshold": 2},
) -> np.ndarray:
    """

    Parameters
    ----------
    src_img :
    dest_img :
    mp_kwargs :
    orb_kwargs :
    ransac_kwargs :

    Returns
    -------

    """
    # Calculate Mean Phase Image
    src_mp = np.log(ep.preprocess.mean_phase(src_img, **mp_kwargs))
    dest_mp = np.log(ep.preprocess.mean_phase(dest_img, **mp_kwargs))

    # Detect Features
    dest_key, dest_feat = orb_detect_extract(dest_mp, **orb_kwargs)
    src_key, src_feat = orb_detect_extract(src_mp, **orb_kwargs)

    # Match Features
    matches = match_descriptors(dest_feat, src_feat, cross_check=True)

    # Compute Affine
    dest_key = dest_key[:, [1, 0]]
    src_key = src_key[:, [1, 0]]

    model_robust, inliers = ransac(
        (src_key[matches[:, 1]], dest_key[matches[:, 0]]),
        AffineTransform,
        min_samples=3, **ransac_kwargs
    )

    # Warp Image
    transformed_src = warp(src_img, model_robust.inverse)

    return transformed_src
