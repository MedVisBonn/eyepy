# -*- coding: utf-8 -*-
from .loggabor import mean_phase


def center_crop(img, size, no_zero_pad=False):
    center = img.shape[0] // 2, img.shape[1] // 2

    height_range = center[0] - size[0] // 2, center[0] + size[0] // 2
    width_range = center[1] - size[1] // 2, center[1] + size[1] // 2
    return img[height_range[0] : height_range[1], width_range[0] : width_range[1], :]
