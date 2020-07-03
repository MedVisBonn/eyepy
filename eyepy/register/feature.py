# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import hog, ORB, daisy
from skimage.filters import gaussian
from annoy import AnnoyIndex


def orb_extract(img, **kwargs):
    descriptor_extractor = ORB(**kwargs)
    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def mind_extract(img, r=0, sigma=0.5):
    mind_features = mind(img, r, sigma)
    features = []
    keys = []
    for key in np.ndindex(img.shape[:2]):
        feature = mind_features[key]
        keys.append(key)
        features.append(feature)

    return np.array(keys), np.array(features)


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


def daisy_extract(img, step=4, radius=15, ring_radii=None, **daisy_kwargs):
    daisy_features = daisy(img, step, radius, ring_radii=ring_radii, **daisy_kwargs)

    keys = []
    features = []
    for key in np.ndindex(daisy_features.shape[:-1]):
        feature = daisy_features[key]
        keys.append(key)
        features.append(feature)

    features = np.array(features)
    keys = np.array(keys)

    # Rescale features to original image positions
    if ring_radii:
        radius = ring_radii[-1]
    keys = keys * step + radius

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


def _match_ann(feat1, feat2, n_trees):
    t = AnnoyIndex(feat1.shape[-1], "euclidean")
    for i in range(feat1.shape[0]):
        t.add_item(i, feat1[i])

    t.build(n_trees)

    matches = np.empty((feat2.shape[0], 2), dtype=np.int)
    for i in range(feat2.shape[0]):
        ind = t.get_nns_by_vector(feat2[i], n=1, search_k=-1)

        matches[i, :] = np.array([i, ind[0]])

    return matches


def match_ann_descriptors(feat1, feat2, n_trees=10, cross_check=True):
    matches1 = _match_ann(feat1, feat2, n_trees)

    if cross_check:
        matches2 = _match_ann(feat2, feat1, n_trees)

        matches1 = {tuple(i) for i in matches1}
        matches2 = {tuple(i) for i in matches2[:, [1, 0]]}
        matches = matches1.intersection(matches2)

        matches1 = np.array(list(matches))

    return matches1[:, [1, 0]]


def search_region(r):
    """
    >>> search_region(1)
    (array([-1, -1, -1,  0,  0,  1,  1,  1]), array([-1,  0,  1, -1,  1, -1,  0,  1]))
    >>> search_region(0)
    (array([ 1, -1,  0,  0]), array([ 0,  0,  1, -1]))
    """
    if r > 0:  # dense sampling with half-width r
        xs, ys = np.mgrid[-r : r + 1, -r : r + 1]
        xs = xs.flatten()
        ys = ys.flatten()
        mid = int((len(xs) + 1) / 2)
        xs = xs[np.arange(len(xs)) != mid - 1]
        ys = ys[np.arange(len(ys)) != mid - 1]
    else:  # six-neighbourhood
        xs = np.array([1, -1, 0, 0])
        ys = np.array([0, 0, 1, -1])

    return xs, ys


def imgshift(img, x, y):
    """
    >>> img = np.arange(0,9).reshape(3,3)
    >>> imgshift(img, 1,1)
    array([[4, 5, 2],
           [7, 8, 5],
           [6, 7, 8]])

    >>> imgshift(img, 1,-1)
    array([[0, 1, 2],
           [1, 2, 5],
           [4, 5, 8]])
    """
    rows, cols = img.shape[0], img.shape[1]
    imshift = img.copy()

    x1s = max(0, x)
    x2s = min(cols, cols + x)
    #

    y1s = max(0, y)
    y2s = min(rows, rows + y)

    x1 = max(0, -x)
    x2 = min(cols, cols - x)

    y1 = max(0, -y)
    y2 = min(rows, rows - y)

    imshift[y1:y2, x1:x2] = img[y1s:y2s, x1s:x2s]
    return imshift


def mind(img, r=0, sigma=0.5):
    """ Calculation of MIND (modality independent neighbourhood descriptor)

    This is not the reference implementation but should yield the same results.
    You can find the reference implementation here: http://www.mpheinrich.de/software.html

    If you use this implementation please cite:
    M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
    Descriptor for Multi-Modal Deformable Registration"
    Medical Image Analysis (2012)

    img : Image
    r : half-width of spatial search (large values may cause: "out of memory")
        r=0 uses a six-neighbourhood, other dense sampling
    sigma : Gaussian weighting for patches
    """

    img = img.astype(np.float32)

    xs, ys = search_region(r)
    xs0, ys0 = search_region(0)

    dist_p = np.zeros(img.shape + (len(xs0),), dtype=np.float32)

    # Calculating Gaussian weighted patch SSD using convolution
    # Store the smoothed squared difference between image and shifted image
    for i, (x, y) in enumerate(zip(xs0, ys0)):
        dist_p[..., i] = gaussian(
            (img - imgshift(img, x, y)) ** 2, sigma=sigma, multichannel=True
        )

    # Variance measure for Gaussian function
    variance = np.mean(dist_p, axis=-1)

    # the following can improve robustness
    # (by limiting V to be in smaller range)
    min_val = 0.001 * np.mean(variance)
    max_val = 1000.0 * np.mean(variance)
    variance = np.minimum(np.maximum(variance, min_val), max_val)

    img1 = np.zeros(img.shape + (len(xs0),), dtype=np.float32)
    for i in range(len(xs0)):
        img1[..., i] = np.exp(-dist_p[..., i] / variance)

    mind_desc = np.zeros(img.shape + (len(xs),), dtype=np.float32)
    # descriptor calculation according to Eq. 4
    if r > 0:
        for i, (x, y) in enumerate(zip(xs, ys)):
            mind_desc[..., i] = np.exp(
                -gaussian(
                    (img - imgshift(img, x, y)) ** 2, sigma=sigma, multichannel=True
                )
                / variance
            )
    else:
        # if six-neighbourhood is used, all patch distances are already calculated
        mind_desc = img1.copy()

    # normalise descriptors to a maximum/mean of 1
    mind_max = np.amax(mind_desc, axis=-1)
    mind_desc = mind_desc / mind_max[..., np.newaxis]

    return mind_desc
