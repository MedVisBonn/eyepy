"""Hardcoded quantification grid presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridPreset:
    """Standard quantification grid definition with fixed region names.

    ``region_names`` are ordered to match mask generation in
    :func:`~eyepy.core.grids.grid`.
    """

    name: str
    radii: tuple[float, ...]
    n_sectors: tuple[int, ...]
    offsets: tuple[float, ...]
    region_names: tuple[str, ...]


ETDRS_9 = GridPreset(
    name='ETDRS_9',
    radii=(0.5, 1.5, 3.0),
    n_sectors=(1, 4, 4),
    offsets=(0, 45, 45),
    region_names=(
        'Central',
        'Inner Superior',
        'Inner Nasal',
        'Inner Inferior',
        'Inner Temporal',
        'Outer Superior',
        'Outer Nasal',
        'Outer Inferior',
        'Outer Temporal',
    ),
)
