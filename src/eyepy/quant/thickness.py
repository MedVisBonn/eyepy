"""Layer and voxel-annotation thickness calculations."""

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def thickness_from_layer_pair(
    top: npt.NDArray,
    bottom: npt.NDArray,
    scale_y: float,
) -> npt.NDArray:
    """Per-A-scan retinal thickness in physical units (scale_unit).

    Result unit = unit of scale_y (inferred from volume.scale_unit at call site).
    NaN where either boundary is NaN.

    Args:
        top: Upper layer height map, shape (n_bscans, width).
        bottom: Lower layer height map, shape (n_bscans, width).
        scale_y: Vertical scale in physical units per pixel.

    Returns:
        Thickness map with the same shape as the input layer maps.
    """
    if top.shape != bottom.shape:
        raise ValueError(
            f'Layer shape mismatch: top {top.shape} vs bottom {bottom.shape}'
        )

    thickness = (bottom - top) * scale_y
    negative = np.isfinite(thickness) & (thickness < 0)
    if np.any(negative):
        logger.warning(
            'Negative layer-pair thickness at %d positions; check segmentation.',
            int(np.sum(negative)),
        )
    return thickness


def thickness_from_voxel_annotation(
    mask: npt.NDArray,
    scale_y: float,
) -> npt.NDArray:
    """Per-A-scan axial extent of a binary 3D annotation in physical units.

    Sums along axis=1 (A-scan depth), then multiplies by scale_y.
    Shape: (n_bscans, width) — same as EyeVolumePixelAnnotation.projection
    before the flip/warp steps.

    Args:
        mask: Binary 3D annotation, shape (n_bscans, height, width).
        scale_y: Vertical scale in physical units per pixel.

    Returns:
        Per-column thickness map, shape (n_bscans, width).
    """
    return np.nansum(mask, axis=1) * scale_y
