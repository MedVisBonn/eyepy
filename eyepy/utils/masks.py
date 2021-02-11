# -*- coding: utf-8 -*-
import cmath
import logging

import numpy as np
from skimage import transform

from eyepy.utils.filters import radius_filtergrid, theta_filtergrid

logger = logging.getLogger(__name__)


def circle_mask(radius, mask_shape=None, smooth_edges=False):
    """Create a centered circular mask with given radius.

    Parameters
    ----------
    radius :
    mask_shape :
    smooth_edges :

    Returns
    -------
    """
    if mask_shape is None:
        mask_shape = (radius * 2, radius * 2)

    if smooth_edges:
        work_shape = (mask_shape[0] * 5, mask_shape[1] * 5)
        radius *= 5
    else:
        work_shape = mask_shape

    circle_mask = np.zeros(work_shape)
    circle_mask[
        radius_filtergrid(work_shape, quadrant_shift=False, normalize=False) < radius
    ] = 1

    return transform.resize(circle_mask, mask_shape)


def sector_masks(mask_shape, n_sectors=4, sector_rotations=0, smooth_edges=False):
    """Create masks for n radial sectors.

    By default the first sector is the first quadrant, and the remaining 3
    sectors are added counter clockwise.

    For a binary mask pixels can not belong to two mask without changing the
    sum over all masks. But for pixels at the sector edges it is not clear to
    which sector they belong and assigning them to two sectors partially might
    be desired. Hence if smooth_edges is True, we create 5 times bigger binary
    masks and then use the anti-aliasing from downscaling them to the desired
    shape to create a float mask.

    Parameters
    ----------
    mask_shape :
    n_sectors :
    sector_rotations :
    smooth_edges :

    Returns
    -------
    """
    if smooth_edges:
        work_shape = (mask_shape[0] * 5, mask_shape[1] * 5)
    else:
        work_shape = mask_shape

    theta = theta_filtergrid(work_shape, quadrant_shift=False)
    # Convert from angles in radian range [-pi, +pi] to degree range [0, 360]
    theta = theta / np.pi * 180
    theta[np.where(theta < 0)] += 360

    masks = []
    sector_size = 360 / n_sectors
    for i in range(n_sectors):
        sector_start = i * sector_size + sector_rotations
        sector_end = (i + 1) * sector_size + sector_rotations

        if sector_end >= 360 and sector_start >= 360:
            sector_end -= 360
            sector_start -= 360

        if sector_end >= 360:
            mask = np.ones(work_shape)
            mask[
                np.where(
                    np.logical_and(theta < sector_start, theta >= sector_end - 360)
                )
            ] = 0
        else:
            mask = np.zeros(work_shape)
            mask[
                np.where(np.logical_and(theta >= sector_start, theta < sector_end))
            ] = 1

        if smooth_edges:
            mask = transform.resize(mask, mask_shape)

        masks.append(mask)

    return masks


def create_region_shape_primitives(
    mask_shape,
    radii: list = (0.8, 1.8),
    n_sectors: list = (1, 4),
    rotation: list = (0, 45),
    center=None,
):
    """Create circles and lines indicating region boundaries of quantification
    masks. These can used for plotting the masks.

    Parameters
    ----------
    mask_shape :
    radii :
    n_sectors :
    rotation :
    center :

    Returns
    -------
    """
    if center is None:
        center = (mask_shape[0] / 2, mask_shape[0] / 2)

    primitives = {"circles": [], "lines": []}
    # Create circles
    for radius in radii:
        primitives["circles"].append({"center": center, "radius": radius})

    for i, (n_sec, rot, radius) in enumerate(zip(n_sectors, rotation, radii)):
        rot = rot / 360 * 2 * np.pi
        if not n_sec is None and n_sec != 1:
            for sec in range(n_sec):
                theta = 2 * np.pi / n_sec * sec + rot

                start = cmath.rect(radii[i - 1], theta)
                start = (start.real + center[0], start.imag + center[1])

                end = cmath.rect(radius, theta)
                end = (end.real + center[0], end.imag + center[1])

                primitives["lines"].append({"start": start, "end": end})

    return primitives


def create_region_masks(
    mask_shape,
    radii: list = (0.8, 1.8),
    n_sectors: list = (1, 4),
    rotation: list = (0, 45),
    center=None,
    smooth_edges=False,
    ring_sectors=True,
    add_circle_masks=False,
) -> list:
    """Create segmented circular region masks for quantification.

    First circular masks with the provided radii are generated. Then ring masks
    are created by subtracting the first circular mask from the second and so
    on.
    Ring masks are only created if they have a respective entry in n_sectors.
    If you want the complete ring, set the respective n_sectors entry to 1. You
    can skip a ring by setting the respective entry to None and you  can split
    the ring into n sectors by setting the respective entry to n.
    Setting a number in `rotation` rotates the respective rings sectors by n
    degree counterclockwise.

    Parameters
    ----------
    mask_shape :
    radii :
    n_sectors :
    rotation :
    center :
    smooth_edges :
    ring_sectors : If False, sectors are not applied to rings but to the complete circles
    add_circle_masks :

    Returns
    -------
    """

    # Create circles
    circles = []
    for radius in radii:
        circles.append(circle_mask(radius, mask_shape, smooth_edges))

    n_sectors = [
        n_sectors[i] if i < len(n_sectors) else None for i, _ in enumerate(radii[1:])
    ]
    rotation = [
        rotation[i] if i < len(rotation) else 0 for i, _ in enumerate(radii[1:])
    ]

    level_sector_parts = []
    for n_sec, rot in zip(n_sectors, rotation):
        if n_sec is not None:
            level_sector_parts.append(
                sector_masks(
                    mask_shape,
                    n_sectors=n_sec,
                    sector_rotations=rot,
                    smooth_edges=smooth_edges,
                )
            )

    if ring_sectors:
        rings = []
        for i, _ in enumerate(circles):
            if i + 1 >= len(circles):
                break
            elif not radii[i] < radii[i + 1]:
                break
            else:
                rings.append(-circles[i] + circles[i + 1])

        pairs = zip(rings, level_sector_parts)
    else:
        pairs = zip(circles, level_sector_parts)

    all_masks = []
    for cir, sectors in pairs:
        for sec in sectors:
            all_masks.append(cir * sec)

    if add_circle_masks:
        [all_masks.append(cir) for cir in circles]

    if center is not None:
        translation = transform.AffineTransform(
            translation=np.array(center) - np.array(mask_shape) / 2
        )
        all_masks = [transform.warp(mask, translation.inverse) for mask in all_masks]

    return all_masks
