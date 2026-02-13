"""Quantification module for ophthalmic image analysis.

This module provides tools for quantifying features in ophthalmic
images, including area measurements and spatial extent calculations
relative to anatomical landmarks.
"""

from eyepy.quant.drusen import drusen
from eyepy.quant.drusen import ideal_rpe
from eyepy.quant.metrics import compute_area
from eyepy.quant.spatial import AnatomicalOrigin
from eyepy.quant.spatial import DirectionalExtent
from eyepy.quant.spatial import ExtentMetrics
from eyepy.quant.spatial import OriginMode
from eyepy.quant.spatial import OriginModeType
from eyepy.quant.spatial import PolarReference

__all__ = [
    'compute_area',
    'drusen',
    'ideal_rpe',
    'AnatomicalOrigin',
    'DirectionalExtent',
    'ExtentMetrics',
    'OriginMode',
    'OriginModeType',
    'PolarReference',
]
