"""Quantification module for ophthalmic image analysis.

This module provides tools for quantifying features in ophthalmic
images, including area measurements and spatial extent calculations
relative to anatomical landmarks.
"""

from eyepy.quant.enface_scalar import EnfaceScalarQuantification
from eyepy.quant.grid import quantize_on_grid
from eyepy.quant.grid_presets import ETDRS_9
from eyepy.quant.grid_presets import GridPreset
from eyepy.quant.metrics import compute_area
from eyepy.quant.spatial import AnatomicalOrigin
from eyepy.quant.spatial import DirectionalExtent
from eyepy.quant.spatial import ExtentMetrics
from eyepy.quant.spatial import OriginMode
from eyepy.quant.spatial import OriginModeType
from eyepy.quant.spatial import PolarReference
from eyepy.quant.thickness import thickness_from_layer_pair
from eyepy.quant.thickness import thickness_from_voxel_annotation

__all__ = [
    'compute_area',
    'EnfaceScalarQuantification',
    'ETDRS_9',
    'GridPreset',
    'quantize_on_grid',
    'thickness_from_layer_pair',
    'thickness_from_voxel_annotation',
    'AnatomicalOrigin',
    'DirectionalExtent',
    'ExtentMetrics',
    'OriginMode',
    'OriginModeType',
    'PolarReference',
]
