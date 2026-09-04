"""Enface scalar map quantification (e.g. layer thickness)."""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, TYPE_CHECKING, Union

from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt

from eyepy.quant.enface import warp_bscan_to_enface
from eyepy.quant.grid import make_quantification_grid
from eyepy.quant.grid import quantize_on_grid
from eyepy.quant.grid import resolve_grid_config
from eyepy.quant.thickness import thickness_from_layer_pair
from eyepy.quant.thickness import thickness_from_voxel_annotation

if TYPE_CHECKING:
    import matplotlib as mpl

    from eyepy.core.eyevolume import EyeVolume
    from eyepy.quant.grid_presets import GridPreset


@dataclass
class EnfaceScalarQuantification:
    """ETDRS quantification of a scalar field on the enface/localizer plane.

    Attributes:
        volume: Source EyeVolume.
        data: Scalar map in enface/localizer coordinates.
        unit: Physical unit from ``volume.scale_unit``.
        radii: ETDRS grid radii.
        n_sectors: Sector counts per ring.
        offsets: Sector rotation offsets in degrees.
        center: Optional grid center in localizer coordinates.
        grid_preset: Standard grid preset with predefined region names.
        names: Explicit region names overriding preset/index naming.
    """

    volume: EyeVolume
    data: npt.NDArray
    unit: str
    radii: Iterable[float] = (1.5, 2.5)
    n_sectors: Iterable[int] = (1, 4)
    offsets: Iterable[int] = (0, 45)
    center: Optional[tuple[float, float]] = None
    grid_preset: Optional[GridPreset] = None
    names: Optional[Sequence[str]] = None
    _masks: Optional[dict[str, npt.NDArray]] = field(default=None, repr=False)
    _quantification: Optional[dict[str, Union[float, str]]] = field(
        default=None,
        repr=False,
    )

    @classmethod
    def from_layer_pair(
        cls,
        volume: EyeVolume,
        top: str,
        bottom: str,
        radii: Iterable[float] = (1.5, 2.5),
        n_sectors: Iterable[int] = (1, 4),
        offsets: Iterable[int] = (0, 45),
        center: Optional[tuple[float, float]] = None,
        names: Optional[Sequence[str]] = None,
        grid: Optional[GridPreset] = None,
    ) -> EnfaceScalarQuantification:
        """Compute thickness between two segmented layer boundaries."""
        if top not in volume.layers:
            raise KeyError(f'Layer {top!r} not found in volume.layers')
        if bottom not in volume.layers:
            raise KeyError(f'Layer {bottom!r} not found in volume.layers')

        grid_config = resolve_grid_config(
            radii=radii,
            n_sectors=n_sectors,
            offsets=offsets,
            names=names,
            grid=grid,
        )

        bscan_map = thickness_from_layer_pair(
            volume.layers[top].data,
            volume.layers[bottom].data,
            volume.scale_y,
        )
        enface_map = warp_bscan_to_enface(
            bscan_map,
            volume.localizer_transform,
            output_shape=volume.localizer.shape,
            order=1,
        )
        return cls(
            volume=volume,
            data=enface_map,
            unit=volume.scale_unit,
            radii=grid_config['radii'],
            n_sectors=grid_config['n_sectors'],
            offsets=grid_config['offsets'],
            center=center,
            grid_preset=grid_config['grid_preset'],
            names=grid_config['names'],
        )

    @classmethod
    def from_pixel_annotation(
        cls,
        volume: EyeVolume,
        name: str,
        radii: Iterable[float] = (1.5, 2.5),
        n_sectors: Iterable[int] = (1, 4),
        offsets: Iterable[int] = (0, 45),
        center: Optional[tuple[float, float]] = None,
        names: Optional[Sequence[str]] = None,
        grid: Optional[GridPreset] = None,
    ) -> EnfaceScalarQuantification:
        """Compute mean axial extent of a binary 3D voxel annotation."""
        if name not in volume.volume_maps:
            raise KeyError(f'Volume map {name!r} not found in volume.volume_maps')

        grid_config = resolve_grid_config(
            radii=radii,
            n_sectors=n_sectors,
            offsets=offsets,
            names=names,
            grid=grid,
        )

        annotation = volume.volume_maps[name]
        bscan_map = thickness_from_voxel_annotation(
            annotation.data,
            volume.scale_y,
        )
        enface_map = warp_bscan_to_enface(
            bscan_map,
            volume.localizer_transform,
            output_shape=volume.localizer.shape,
            order=1,
        )
        return cls(
            volume=volume,
            data=enface_map,
            unit=volume.scale_unit,
            radii=grid_config['radii'],
            n_sectors=grid_config['n_sectors'],
            offsets=grid_config['offsets'],
            center=center,
            grid_preset=grid_config['grid_preset'],
            names=grid_config['names'],
        )

    @property
    def masks(self) -> dict[str, npt.NDArray]:
        """ETDRS grid masks on the localizer plane."""
        if self._masks is None:
            self._masks = make_quantification_grid(
                self.volume,
                radii=self.radii,
                n_sectors=self.n_sectors,
                offsets=self.offsets,
                center=self.center,
                grid_preset=self.grid_preset,
                names=self.names,
            )
        return self._masks

    @property
    def quantification(self) -> dict[str, Union[float, str]]:
        """Mean thickness per ETDRS zone."""
        if self._quantification is None:
            zone_values = quantize_on_grid(
                self.data,
                self.masks,
                aggregator='mean',
                unit=self.unit,
            )
            zone_values['Laterality'] = self.volume.laterality
            self._quantification = zone_values
        return self._quantification

    def _value_key(self, mask_name: str) -> str:
        return f'{mask_name} [{self.unit}]'

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        region: Union[slice, tuple[slice, slice]] = np.s_[:, :],
        cmap: Union[str, mpl.colors.Colormap] = 'viridis',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar: bool = True,
        alpha: float = 1,
    ) -> None:
        """Plot the enface thickness map."""
        ax = plt.gca() if ax is None else ax
        enface_crop = self.data[region]
        valid = np.isfinite(enface_crop)

        if vmin is None:
            vmin = float(np.nanmin(enface_crop)) if np.any(valid) else 0.0
        if vmax is None:
            vmax = float(np.nanmax(enface_crop)) if np.any(valid) else vmin

        visible = np.zeros(enface_crop.shape)
        visible[valid] = 1

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap),
                cax=cax,
                label=self.unit,
            )

        ax.imshow(
            enface_crop,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def plot_quantification(
        self,
        ax: Optional[plt.Axes] = None,
        region: Union[slice, tuple[slice, slice]] = np.s_[:, :],
        alpha: float = 0.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar: bool = True,
        cmap: Union[str, mpl.colors.Colormap] = 'viridis',
    ) -> None:
        """Plot ETDRS mean-thickness values on the localizer."""
        ax = plt.gca() if ax is None else ax

        mask_img = np.zeros(self.volume.localizer.shape, dtype=float)[region]
        visible = np.zeros_like(mask_img)
        for mask_name in self.masks.keys():
            mask_img += (
                self.masks[mask_name][region] *
                self.quantification[self._value_key(mask_name)]
            )
            visible += self.masks[mask_name][region]

        valid = visible.astype(bool)
        if vmin is None:
            vmin = float(mask_img[valid].min()) if np.any(valid) else 0.0
        if vmax is None:
            vmax = max([float(mask_img.max()), vmin]) if np.any(valid) else vmin

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap),
                cax=cax,
                label=self.unit,
            )

        ax.imshow(
            mask_img,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
