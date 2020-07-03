# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import skimage as ski
from skimage.color import gray2rgb
from skimage.segmentation import checkerboard_level_set


def registration_comparison(ax, img1, img2, square_size=200):
    img1 = ski.color.gray2rgb(ski.img_as_float(img1, force_copy=True), alpha=True)
    img2 = ski.color.gray2rgb(ski.img_as_float(img2, force_copy=True), alpha=True)

    alpha = ski.segmentation.checkerboard_level_set(
        img1.shape[:-1], square_size
    ).astype(float)

    img2[..., -1] = alpha
    ax.imshow(img1)
    ax.imshow(img2)
