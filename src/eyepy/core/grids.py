from __future__ import annotations

import cmath
from collections.abc import Sequence
import functools
import logging
from typing import Any, Iterable, Optional, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
from skimage import transform

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    Shape = Union[int, tuple[int, int]]


def circle_mask(radius: int,
                mask_shape: Optional[tuple[int, int]] = None,
                smooth_edges: bool = False) -> npt.NDArray[Any]:
    """Create a centered circular mask with given radius.

    Args:
        radius:
        mask_shape:
        smooth_edges:

    Returns:
    """
    if mask_shape is None:
        mask_shape = (radius * 2, radius * 2)

    if smooth_edges:
        work_shape = (mask_shape[0] * 5, mask_shape[1] * 5)
        radius *= 5
    else:
        work_shape = mask_shape

    circle_mask = np.zeros(work_shape)
    circle_mask[radius_filtergrid(
        work_shape, quadrant_shift=False, normalize=False) < radius] = 1

    return transform.resize(circle_mask, mask_shape)


def create_sectors(mask_shape: tuple[int, int],
                   n_sectors: int = 4,
                   start_angle: int = 0,
                   clockwise: bool = False,
                   smooth_edges: bool = False) -> list[npt.NDArray[Any]]:
    """Create masks for n radial sectors.

    By default the first sector is the first quadrant, and the remaining 3
    sectors are added counter clockwise.

    For a binary mask pixels can not belong to two mask without changing the
    sum over all masks. But for pixels at the sector edges it is not clear to
    which sector they belong and assigning them to two sectors partially might
    be desired. Hence if smooth_edges is True, we create 5 times bigger binary
    masks and then use the anti-aliasing from downscaling them to the desired
    shape to create a float mask.

    Args:
        mask_shape:
        n_sectors:
        start_angle:
        clockwise:
        smooth_edges:

    Returns:
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
        if clockwise:
            theta = np.flip(theta, axis=1)
            sector_start = start_angle - i * sector_size
            sector_end = start_angle - (i + 1) * sector_size
        else:
            sector_start = start_angle + i * sector_size
            sector_end = start_angle + (i + 1) * sector_size

        sector_start = sector_start % 360
        sector_end = sector_end % 360

        mask = np.zeros(work_shape)
        # Handle clockwise and counter-clockwise sector rotation
        if clockwise:

            if sector_start > sector_end:
                # If rotating clockwise the start angle is bigger than the end angle
                selection = np.where(
                    np.logical_and(theta <= sector_start, theta > sector_end))
            else:
                # If this is not the case, only the end angle has crossed the 0°
                selection = np.where(
                    np.logical_or(theta <= sector_start, theta > sector_end))
        else:
            if sector_start < sector_end:
                # If rotating counter-clockwise the start angle is smaller than the end
                selection = np.where(
                    np.logical_and(theta >= sector_start, theta < sector_end))
            else:
                # If this is not the case only the end angle has crossed the 360°
                selection = np.where(
                    np.logical_or(theta >= sector_start, theta < sector_end))

        mask[selection] = 1

        if smooth_edges:
            mask = transform.resize(mask, mask_shape)

        masks.append(mask)

    return masks


# Cache created grids. When quantifying many volumes you need the grid for both laterality and possibly different localizer sizes.
@functools.lru_cache(8, typed=False)
def create_grid_regions(
    mask_shape: tuple[int, int],
    radii: Sequence[int],
    n_sectors: Sequence[int],
    offsets: Sequence[int],
    clockwise: bool,
    smooth_edges: bool = False,
) -> list[npt.NDArray[Any]]:
    """Create sectorized circular region masks.

    First circular masks with the provided radii are generated. Then ring masks
    are created by subtracting the first circular mask from the second and so
    on.
    If you want the complete ring, set the respective n_sectors entry to 1. You  can split
    the ring into n sectors by setting the respective entry to n.
    Setting a number in `offsets` rotates the respective ring sectors by n
    degree.

    Args:
        mask_shape: Output shape of the computed masks
        radii: Ascending radii of the circular regions in pixels
        n_sectors: Number of sectors corresponding to the radii
        offsets: Angular offset of first sector corresponding to the radii
        clockwise: If True sectors are added clockwise starting from the start_angles
        smooth_edges: If True, compute non binary masks where edges might be shared between adjacent regions

    Returns:
    """
    # Create circles
    circles = []
    for radius in radii:
        circles.append(circle_mask(radius, mask_shape, smooth_edges))

    level_sector_parts = []
    for n_sec, start_angle in zip(n_sectors, offsets):
        if n_sec is not None:
            level_sector_parts.append(
                create_sectors(
                    mask_shape,
                    n_sectors=n_sec,
                    start_angle=start_angle,
                    clockwise=clockwise,
                    smooth_edges=smooth_edges,
                ))

    rings = [circles[0]]
    for i, _ in enumerate(circles):
        if i + 1 >= len(circles):
            break
        elif not radii[i] < radii[i + 1]:
            break
        else:
            rings.append(-circles[i] + circles[i + 1])

    pairs = zip(rings, level_sector_parts)

    all_masks = []
    for cir, sectors in pairs:
        for sec in sectors:
            all_masks.append(cir * sec)

    return all_masks


def grid(
    mask_shape: tuple[int, int],
    radii: Union[Sequence[Union[int, float]], int, float],
    laterality: str,
    n_sectors: Union[Sequence[Union[int, float]], int, float] = 1,
    offsets: Union[Sequence[Union[int, float]], int, float] = 0,
    center: Optional[tuple] = None,
    smooth_edges: bool = False,
    radii_scale: Union[int, float] = 1,
) -> dict[str, npt.NDArray[Any]]:
    """Create a quantification grid.

    Args:
        mask_shape: Output shape of the computed masks
        radii: Ascending radii of the circular regions in pixels
        laterality: OD/OS depending for which eye to compute the grid
        n_sectors: Number of sectors corresponding to the radii
        offsets: Sector offsets from the horizonal line on the nasal side in degree
        center: Center location of the computed masks
        smooth_edges: If True, compute non binary masks where edges might be shared between adjacent regions
        radii_scale:

    Returns:
    """
    # Make sure radii, n_sectors and offsets are lists even if you get numbers or tuples
    radii = [radii] if isinstance(radii, (int, float)) else list(radii)
    if not sorted(radii) == radii:
        raise ValueError('radii have to be given in ascending order')
    input_radii = radii
    radii = [r / radii_scale for r in radii]

    n_sectors = [n_sectors] if isinstance(n_sectors,
                                          (int, float)) else list(n_sectors)
    if len(n_sectors) == 1:
        n_sectors = n_sectors * len(radii)

    offsets = [offsets] if isinstance(offsets, (int, float)) else list(offsets)
    if len(offsets) == 1:
        offsets = offsets * len(radii)

    clockwise = False
    masks = create_grid_regions(
        mask_shape,
        tuple(radii),
        tuple(n_sectors),
        tuple(offsets),
        clockwise,
        smooth_edges,
    )

    names = []
    radii = [0.0] + radii
    input_radii = [0] + input_radii
    for i, r in enumerate(radii):
        if i + 1 >= len(radii):
            break
        for s in range(n_sectors[i]):
            names.append(
                f'Radius: {input_radii[i]}-{input_radii[i+1]} Sector: {s}')

    masks = {name: mask for name, mask in zip(names, masks)}
    if laterality == 'OS':
        masks = {name: np.flip(m, axis=1) for name, m in masks.items()}
    elif laterality == 'OD':
        pass
    else:
        raise ValueError('laterality has to be one of OD/OS')

    if center is not None:
        translation = transform.AffineTransform(translation=np.array(center) -
                                                np.array(mask_shape) / 2)
        masks = {
            name: transform.warp(masks[name], translation.inverse)
            for name in masks.keys()
        }

    return masks


@functools.lru_cache(maxsize=8)
def filtergrid(size: Shape,
               quadrant_shift: bool = True,
               normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Generates grid for constructing frequency domain filters.

    Coordinate matrices for x and y value for a 2D array. The out can be quadrant shifted
    and / or normalized. This is basically a wrapper around np.meshgrid.

    Inspired by filtergrid.m found at https://www.peterkovesi.com/matlabfns/

    Args:
        size: Size of the filter
        quadrant_shift: Quadrant shift such that 0 values / frequencies are at the corners
        normalize: Normalize the range to [-0.5,0.5]

    Returns:
    """
    if type(size) is int:
        rows = cols = size
    else:
        rows = size[0]
        cols = size[1]

    range_1 = np.linspace(-(cols // 2), np.floor((cols - 1) / 2), cols)
    range_2 = np.linspace(-(rows // 2), np.floor((rows - 1) / 2), rows)

    if normalize:
        range_1 = range_1 / cols
        range_2 = range_2 / rows

    x, y = np.meshgrid(range_1, range_2)

    # Quadrant shift so that filters are constructed with 0 frequency at the corners
    if quadrant_shift:
        x = np.fft.ifftshift(x)
        y = np.fft.ifftshift(y)

    return x.T, y.T


@functools.lru_cache(maxsize=8)
def radius_filtergrid(size: Shape,
                      quadrant_shift: bool = True,
                      normalize: bool = True) -> np.ndarray:
    """Radius Filtergrid.

    A matrix containing the radius from the center. This radius is in range [0, 0.5] if normalized.
    The result can be quadrant shifted such that the 0 values are in the corners.

    Args:
        size: Size of the filter
        quadrant_shift: Quadrant shift such that 0 values / frequencies are at the corners
        normalize: Normalize radius to [0 ,0.5]

    Returns:
    """
    x, y = filtergrid(size, quadrant_shift, normalize)
    radius = np.sqrt(x**2 + y**2)
    return radius


@functools.lru_cache(maxsize=8)
def theta_filtergrid(size: Shape, quadrant_shift: bool = True) -> np.ndarray:
    """Theta Filtergrid.

    A matrix containing the polar angle in radian at the respective position for a circle centered in the matrix.
    The result can be returned quadrant shifted. The angle is 0 for all points on the positive x-axis.
    The angles are pi/2 (90°) and -pi/2 (-90°) on the positive and negative y-axis respectively. On the negative
    x-axis the angle is pi (180°). If you need the angle to be in range [0, 2pi] instead of [-pi, pi], you can simply
    add 2pi whenever the angle is negative.

    Args:
        size: Size of the filter
        quadrant_shift: Quadrant shift such that 0 values / frequencies are at the corners

    Returns:
    """
    y, x = filtergrid(size, quadrant_shift)

    # Matrix values contain polar angle.
    # 0 angle starts on the horizontal line and runs counter clock-wise
    theta = np.arctan2(-y, x)

    return theta
