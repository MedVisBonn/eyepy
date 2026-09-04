"""ETDRS grid quantification utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from eyepy.quant.metrics import masked_mean
from eyepy.quant.metrics import masked_median

if TYPE_CHECKING:
    from eyepy.quant.grid_presets import GridPreset

Aggregator = Literal['sum', 'mean', 'median']


def resolve_grid_config(
    *,
    radii: Sequence[float] = (1.5, 2.5),
    n_sectors: Sequence[int] = (1, 4),
    offsets: Sequence[float] = (0, 45),
    names: Optional[Sequence[str]] = None,
    grid_preset: Optional[GridPreset] = None,
    grid: Optional[GridPreset] = None,
) -> dict:
    """Resolve grid parameters, expanding a preset shorthand when provided."""
    preset = grid_preset if grid_preset is not None else grid
    if preset is not None:
        return {
            'radii': preset.radii,
            'n_sectors': preset.n_sectors,
            'offsets': preset.offsets,
            'names': names,
            'grid_preset': preset,
        }

    return {
        'radii': radii,
        'n_sectors': n_sectors,
        'offsets': offsets,
        'names': names,
        'grid_preset': None,
    }


def quantize_on_grid(
    data_map: npt.NDArray,
    masks: dict[str, npt.NDArray],
    aggregator: Aggregator = 'mean',
    unit: str = '',
    scale_factor: float = 1.0,
) -> dict[str, float]:
    """Aggregate a scalar enface map within circular grid regions.

    Args:
        data_map: Scalar field in enface/localizer coordinates.
        masks: Named region masks from ``grids.grid()``.
        aggregator: Regional aggregation method.
        unit: Unit suffix for result keys, e.g. ``'mm'`` or ``'mm³'``.
        scale_factor: Multiplier applied after summation (used for volume).

    Returns:
        Dictionary mapping ``'{zone_name} [{unit}]'`` to aggregated values.
    """
    results: dict[str, float] = {}
    for name, mask in masks.items():
        region_mask = mask.astype(bool)
        if aggregator == 'sum':
            value = float((data_map * mask).sum() * scale_factor)
        elif aggregator == 'mean':
            value = masked_mean(data_map, region_mask)
        elif aggregator == 'median':
            value = masked_median(data_map, region_mask)
        else:
            raise ValueError(f'Unknown aggregator: {aggregator!r}')

        results[f'{name} [{unit}]'] = value

    return results


def make_quantification_grid(
    volume,
    radii=(1.5, 2.5),
    n_sectors=(1, 4),
    offsets=(0, 45),
    center=None,
    names: Optional[Sequence[str]] = None,
    grid_preset: Optional[GridPreset] = None,
) -> dict[str, npt.NDArray]:
    """Build ETDRS-style grid masks for a volume's localizer."""
    from eyepy.core.grids import grid as build_grid

    grid_config = resolve_grid_config(
        radii=radii,
        n_sectors=n_sectors,
        offsets=offsets,
        names=names,
        grid_preset=grid_preset,
    )

    return build_grid(
        mask_shape=volume.localizer.shape,
        radii=grid_config['radii'],
        laterality=volume.laterality,
        n_sectors=grid_config['n_sectors'],
        offsets=grid_config['offsets'],
        radii_scale=volume.scale_x,
        center=center,
        grid_preset=grid_config['grid_preset'],
        names=grid_config['names'],
    )
