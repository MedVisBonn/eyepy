# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import skimage as ski
from skimage.color import gray2rgb
from skimage.segmentation import checkerboard_level_set


def to_rgb_with_alpha(img, float_scale=True):
    if img.min() < 0:
        msg = "Negative values are not allowed in image"
        raise ValueError(msg)

    # Integer data should be scaled 0-255
    if img.dtype == int:
        if img.max() > 255:
            msg = "Images of type int have to be scaled 0 to 255"

    # Float data should be scaled 0-1
    if img.dtype == float:
        if img.max() > 1.0:
            msg = "Images of type float have to be scaled 0.0 to 1.0"
            raise ValueError(msg)

    if float_scale and img.dtype == int:
        img /= 255

    if not float_scale and img.dtype == float:
        img = (img * 255).astype(int)

    img = ski.color.gray2rgb(img, alpha=True)
    return img


def registration_comparison(ax, img1, img2, square_size=200):
    img1 = to_rgb_with_alpha(img1, float_scale=True)
    img2 = to_rgb_with_alpha(img2, float_scale=True)

    alpha = ski.segmentation.checkerboard_level_set(
        img1.shape[:-1], square_size
    ).astype(float)

    img2[..., -1] = alpha
    ax.imshow(img1)
    ax.imshow(img2)
