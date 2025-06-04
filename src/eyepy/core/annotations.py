from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import logging
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt
from skimage import transform

from eyepy import config
import eyepy as ep

if TYPE_CHECKING:
    import matplotlib as mpl

    from eyepy import EyeEnface
    from eyepy import EyeVolume

logger = logging.getLogger('eyepy.core.annotations')

SLAB_PROJECTION_DEFAULTS = {
    'NFLVP': {'PAR': False, 'contrast': 2},
    'SVP': {'PAR': False, 'contrast': 2},
    'ICP': {'PAR': True, 'contrast': 2},
    'DCP': {'PAR': True, 'contrast': 2},
    'SVC': {'PAR': False, 'contrast': 4},
    'DVC': {'PAR': True, 'contrast': 4},
    'AVAC': {'PAR': True, 'contrast': 10}, # Avascular Complex
    'RET': {'PAR': False, 'contrast': 'auto'}, # Retina
}


class EyeVolumeLayerAnnotation:

    def __init__(
        self,
        volume: EyeVolume,
        data: Optional[npt.NDArray[np.float64]] = None,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Layer annotation for a single layer in an EyeVolume.

        Args:
            volume: EyeVolume object
            data: 2D array of shape (n_bscans, bscan_width) holding layer height values
            meta: dict with additional meta data
            **kwargs: additional meta data specified as parameters

        Returns:
            None
        """
        self.volume = volume
        if data is None:
            self.data = np.full((volume.size_z, volume.size_x), np.nan)
        else:
            self.data = data

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        # knots is a dict layername: list of curves where every curve is a list of knots
        if 'knots' not in self.meta:
            self.meta['knots'] = defaultdict(lambda: [])
        elif type(self.meta['knots']) is dict:
            self.meta['knots'] = defaultdict(lambda: [], self.meta['knots'])

        if 'name' not in self.meta:
            self.meta['name'] = 'Layer Annotation'

        self.meta['current_color'] = config.layer_colors[self.name]

    @property
    def name(self) -> str:
        """Layer name."""
        return self.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.meta['name'] = value

    @property
    def knots(self) -> dict:
        """Knots parameterizing the layer."""
        return self.meta['knots']

    def layer_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns pixel indices of the layer in the volume.

        While the layer is stored as the offset from the bottom of the OCT volume, some applications require
        layer discretized to voxel positions. This method returns the layer as indices into the OCT volume.

        The indices can be used for example to create layer maps for semantic segmentation.

        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        import eyepy as ep

        eye_volume = ep.data.load("drusen_patient")
        rpe_annotation = eye_volume.layers["RPE"]
        rpe_indices = rpe_annotation.layer_indices()
        rpe_map = np.zeros(eye_volume.shape)
        rpe_map[rpe_indices] = 1
        plt.imshow(rpe_map[0]) # (1)
        ```

        1.  Visualize layer map for the first B-scan

        Returns:
            A tuple with indices for the layers position in the volume - Tuple[bscan_indices, row_indices, column_indices]
        """
        layer = self.data[:, np.newaxis, :]
        nan_indices = np.isnan(layer)
        row_indices = np.rint(layer).astype(int)[~nan_indices]
        x = np.ones(layer.shape)
        x[nan_indices] = 0
        bscan_indices, _, col_indices = np.nonzero(x)
        return (bscan_indices, row_indices, col_indices)


class EyeVolumePixelAnnotation:

    def __init__(
        self,
        volume: EyeVolume,
        # Type hint for an optional boolean numpy array
        data: Optional[npt.NDArray[np.bool_]] = None,
        meta: Optional[dict] = None,
        radii: Iterable[float] = (1.5, 2.5),
        n_sectors: Iterable[int] = (1, 4),
        offsets: Iterable[int] = (0, 45),
        center: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Pixel annotation for an EyeVolume.

        Args:
            volume: EyeVolume object
            data: 3D array of shape (n_bscans, bscan_height, bscan_width) holding boolean pixel annotations
            meta: dict with additional meta data
            radii: radii for quantification on circular grid
            n_sectors: number of sectors for quantification on circular grid
            offsets: offsets from x axis for first sector, for quantification on circular grid
            center: center of circular grid for quantification
            **kwargs: additional meta data specified as parameters

        Returns:
            None
        """
        self.volume = volume

        if data is None:
            self.data = np.full(self.volume.shape,
                                fill_value=False,
                                dtype=bool)
        else:
            self.data = data

        self._masks = None
        self._quantification = None

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        self.meta.update(
            **{
                'radii': radii,
                'n_sectors': n_sectors,
                'offsets': offsets,
                'center': center,
            })

        if 'name' not in self.meta:
            self.meta['name'] = 'Voxel Annotation'

    @property
    def name(self) -> str:
        """Annotation name."""
        return self.meta['name']

    @name.setter
    def name(self, value) -> None:
        self.meta['name'] = value

    def _reset(self) -> None:
        self._masks = None
        self._quantification = None

    @property
    def radii(self) -> Iterable[float]:
        """Radii for quantification on circular grid."""
        return self.meta['radii']

    @radii.setter
    def radii(self, value: Iterable[float]) -> None:
        self._reset()
        self.meta['radii'] = value

    @property
    def n_sectors(self) -> Iterable[int]:
        """Number of sectors for quantification on circular grid."""
        return self.meta['n_sectors']

    @n_sectors.setter
    def n_sectors(self, value: Iterable[int]) -> None:
        self._reset()
        self.meta['n_sectors'] = value

    @property
    def offsets(self) -> Iterable[int]:
        """Offsets from x axis for first sector, for quantification on circular
        grid."""
        return self.meta['offsets']

    @offsets.setter
    def offsets(self, value: Iterable[int]) -> None:
        self._reset()
        self.meta['offsets'] = value

    @property
    def center(self) -> tuple[float, float]:
        """Center of circular grid for quantification."""
        return self.meta['center']

    @center.setter
    def center(self, value: tuple[float, float]) -> None:
        self._reset()
        self.meta['center'] = value

    @property
    def projection(self) -> np.ndarray:
        """Projection of the annotation to the enface plane."""
        # The flip is required because in the volume the bottom most B-scan has the lowest index
        # while in the enface projection the bottom most position should have the biggest index.
        return np.flip(np.nansum(self.data, axis=1), axis=0)

    @property
    def enface(self) -> np.ndarray:
        """Transformed projection of the annotation to the enface plane."""
        return transform.warp(
            self.projection,
            self.volume.localizer_transform.inverse,
            output_shape=(
                self.volume.localizer.size_y,
                self.volume.localizer.size_x,
            ),
            order=0,
            cval=np.nan,
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        region: Union[slice, tuple[slice, slice]] = np.s_[:, :],
        cmap: Union[str, mpl.colors.Colormap] = 'Reds',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar: bool = True,
        alpha: float = 1,
    ) -> None:
        """Plot the annotation on the enface plane.

        Args:
            ax: matplotlib axes object
            region: region of the enface projection to plot
            cmap: colormap
            vmin: minimum value for colorbar
            vmax: maximum value for colorbar
            cbar: whether to plot a colorbar
            alpha: alpha value for the annotation

        Returns:
            None
        """
        enface_projection = self.enface

        ax = plt.gca() if ax is None else ax

        if vmin is None:
            vmin = 1
        if vmax is None:
            vmax = max([enface_projection.max(), vmin])

        enface_crop = enface_projection[region]
        visible = np.zeros(enface_crop.shape)
        visible[np.logical_and(vmin <= enface_crop, enface_crop <= vmax)] = 1

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap),
                cax=cax,
            )

        ax.imshow(
            enface_crop,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Masks for quantification on circular grid.

        Returns:
            A dictionary of masks with the keys being the names of the masks.
        """
        from eyepy.core.grids import grid

        if self._masks is None:
            self._masks = grid(
                mask_shape=self.volume.localizer.shape,
                radii=self.radii,
                laterality=self.volume.laterality,
                n_sectors=self.n_sectors,
                offsets=self.offsets,
                radii_scale=self.volume.scale_x,
                center=self.center,
            )

        return self._masks

    @property
    def quantification(self) -> dict[str, Union[float, str]]:
        """Quantification of the annotation on the specified circular grid.

        Returns:
            A dictionary of quantifications with the keys being the names of the regions.
        """
        if self._quantification is None:
            self._quantification = self._quantify()

        return self._quantification

    def _quantify(self) -> dict[str, Union[float, str]]:
        enface_voxel_size_ym3 = (self.volume.localizer.scale_x * 1e3 *
                                 self.volume.localizer.scale_y * 1e3 *
                                 self.volume.scale_y * 1e3)
        oct_voxel_size_ym3 = (self.volume.scale_x * 1e3 * self.volume.scale_z *
                              1e3 * self.volume.scale_y * 1e3)

        enface_projection = self.enface

        results = {}
        for name, mask in self.masks.items():
            results[f'{name} [mm³]'] = ((enface_projection * mask).sum() *
                                        enface_voxel_size_ym3 / 1e9)

        results['Total [mm³]'] = enface_projection.sum(
        ) * enface_voxel_size_ym3 / 1e9
        results['Total [OCT voxels]'] = self.projection.sum()
        results['OCT Voxel Size [µm³]'] = oct_voxel_size_ym3
        results['Laterality'] = self.volume.laterality
        return results

    # # Todo
    # def create_region_shape_primitives(
    #     mask_shape,
    #     radii: list = (0.8, 1.8),
    #     n_sectors: list = (1, 4),
    #     rotation: list = (0, 45),
    #     center=None,
    # ):
    #     """Create circles and lines indicating region boundaries of quantification
    #     masks. These can be used for plotting the masks.
    #
    #     Parameters
    #     ----------
    #     mask_shape :
    #     radii :
    #     n_sectors :
    #     rotation :
    #     center :
    #
    #     Returns
    #     -------
    #     """
    #     if center is None:
    #         center = (mask_shape[0] / 2, mask_shape[0] / 2)
    #
    #     primitives = {"circles": [], "lines": []}
    #     # Create circles
    #     for radius in radii:
    #         primitives["circles"].append({"center": center, "radius": radius})
    #
    #     for i, (n_sec, rot, radius) in enumerate(zip(n_sectors, rotation, radii)):
    #         rot = rot / 360 * 2 * np.pi
    #         if not n_sec is None and n_sec != 1:
    #             for sec in range(n_sec):
    #                 theta = 2 * np.pi / n_sec * sec + rot
    #
    #                 start = cmath.rect(radii[i - 1], theta)
    #                 start = (start.real + center[0], start.imag + center[1])
    #
    #                 end = cmath.rect(radius, theta)
    #                 end = (end.real + center[0], end.imag + center[1])
    #
    #                 primitives["lines"].append({"start": start, "end": end})
    #
    #     return primitives

    def plot_quantification(
        self,
        ax: Optional[plt.Axes] = None,
        region: Union[slice, tuple[slice, slice]] = np.s_[:, :],
        alpha: float = 0.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar: bool = True,
        cmap: Union[str, mpl.colors.Colormap] = 'YlOrRd',
    ) -> None:
        """Plot circular grid quantification of the annotation (like ETDRS)

        Args:
            ax: Matplotlib axes to plot on
            region: Region to plot
            alpha: Alpha value of the mask
            vmin: Minimum value for the colorbar
            vmax: Maximum value for the colorbar
            cbar: Whether to plot a colorbar
            cmap: Colormap to use

        Returns:
            None
        """

        ax = plt.gca() if ax is None else ax

        mask_img = np.zeros(self.volume.localizer.shape, dtype=float)[region]
        visible = np.zeros_like(mask_img)
        for mask_name in self.masks.keys():
            mask_img += (self.masks[mask_name][region] *
                         self.quantification[mask_name + ' [mm³]'])
            visible += self.masks[mask_name][region]

        vmin = mask_img[visible.astype(int)].min() if vmin is None else vmin
        vmax = max([mask_img.max(), vmin]) if vmax is None else vmax

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap),
                cax=cax,
            )

        ax.imshow(
            mask_img,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )


class EyeVolumeSlabAnnotation:
    """"""

    def __init__(
        self,
        volume: EyeVolume,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            volume: An EyeVolume object
            meta: dict with additional meta data
            **kwargs: additional meta data as keyword arguments
        """
        self.volume = volume
        self._mask = None

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        # Set default values if not provided
        if 'name' not in self.meta:
            self.meta['name'] = 'OCTA Slab'

        # Ensure layer references are present
        if 'top_layer' not in self.meta:
            self.meta['top_layer'] = None
        if 'bottom_layer' not in self.meta:
            self.meta['bottom_layer'] = None

    @property
    def name(self) -> str:
        """Slab name."""
        return self.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.meta['name'] = value

    @property
    def top_layer(self) -> Optional[str]:
        """Top layer name/acronym."""
        return self.meta['top_layer']

    @top_layer.setter
    def top_layer(self, value: str) -> None:
        self.meta['top_layer'] = value

    @property
    def bottom_layer(self) -> Optional[str]:
        """Bottom layer name/acronym."""
        return self.meta['bottom_layer']

    @bottom_layer.setter
    def bottom_layer(self, value: str) -> None:
        self.meta['bottom_layer'] = value

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Mask of the slab in the volume."""
        if self.top_layer is None or self.bottom_layer is None:
            logger.warning('Top or bottom layer not set. Cannot create slab mask.')
            return np.zeros(self.volume.shape, dtype=bool)

        if self.top_layer not in self.volume.layers or self.bottom_layer not in self.volume.layers:
            logger.warning('Top or bottom layer not found in the volume layers. Cannot create slab mask.')
            return np.zeros(self.volume.shape, dtype=bool)

        if self._mask is None:
            top_data = self.volume.layers[self.top_layer].data
            bottom_data = self.volume.layers[self.bottom_layer].data
            self._mask = ep.core.utils.mask_from_boundaries(
                upper=top_data,
                lower=bottom_data,
                height=self.volume.size_y,
            )

        return self._mask

    @property
    def projection(self) -> np.ndarray:
        """Projection of the data within the slab mask to the enface plane."""
        # The flip is required because in the volume the bottom most B-scan has the lowest index
        # while in the enface projection the bottom most position should have the biggest index.
        def get_projection(par: bool = False) -> np.ndarray:
            data = self.volume.data_par if par else self.volume.data
            return np.flip(np.nansum(data * self.mask, axis=1), axis=0)
        return get_projection

    @property
    def enface(self) -> np.ndarray:
        """Transformed projection of the annotation to the enface plane."""
        def get_enface(par: bool = False) -> np.ndarray:
            data = self.projection(par=par)
            return transform.warp(
                data,
                self.volume.localizer_transform.inverse,
                output_shape=(
                    self.volume.localizer.size_y,
                    self.volume.localizer.size_x,
                ),
                order=0,
                cval=np.nan,
            )
        return get_enface

    def iqr_contrast(self, enface: np.ndarray, factor: float=1.5) -> float:
        valid_data = enface[~np.isnan(enface)]
        if len(valid_data) == 0:
            return 1.0

        q75 = np.percentile(valid_data, 75)
        q25 = np.percentile(valid_data, 25)
        iqr = q75 - q25

        return np.round(q75 + factor * iqr)

    def apply_contrast(self, enface: np.ndarray, contrast: float) -> np.ndarray:
        """Apply contrast to the enface projection."""
        if contrast <= 0:
            logger.warning(f'Invalid contrast value: {contrast}. Using default contrast of 4.')
            contrast = 4

        return np.clip(enface / contrast, 0, 1.0)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        region: Union[slice, tuple[slice, slice]] = np.s_[:, :],
        cmap: Union[str, mpl.colors.Colormap] = 'Greys_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar: bool = False,
        alpha: float = 1,
        contrast: Union[int, Literal['auto']] = None,
        par: bool = None,
        transform: bool = False,
        **kwargs
    ) -> None:
        """Plot the annotation on the enface plane.

        Args:
            ax: matplotlib axes object
            region: region of the enface projection to plot
            cmap: colormap
            vmin: minimum value for colorbar
            vmax: maximum value for colorbar
            cbar: whether to plot a colorbar
            alpha: alpha value for the annotation
            contrast: contrast value for the annotation
            par: whether to apply Projection Artifact Removal (PAR) to the enface projection
            transform: whether to apply the localizer transform to the enface projection

        Returns:
            None
        """
        if par is None:
            par = SLAB_PROJECTION_DEFAULTS.get(self.name, {}).get('PAR', False)

        enface_projection = self.enface(par) if transform else self.projection(par)

        if contrast is None:
            contrast = SLAB_PROJECTION_DEFAULTS.get(self.name, {}).get('contrast', 'auto')
        if contrast == 'auto':
            contrast = self.iqr_contrast(self.projection(par), kwargs.get('factor', 1.5))
        elif not isinstance(contrast, (int, float)) or contrast <= 0:
            logger.warning(
                f'Invalid contrast value: {contrast}. Using default contrast of 4.')
            contrast = 4
        else:
            contrast = int(contrast)

        enface_crop = self.apply_contrast(enface_projection[region], contrast)

        ax = plt.gca() if ax is None else ax

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.nanmax([np.nanmax(enface_crop), vmin])

        visible = np.zeros(enface_crop.shape)
        visible[np.logical_and(vmin < enface_crop, enface_crop <= vmax)] = 1

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap),
                cax=cax,
            )

        ax.imshow(
            enface_crop,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )


class EyeBscanLayerAnnotation:

    def __init__(self, eyevolumelayerannotation: EyeVolumeLayerAnnotation,
                 index: int) -> None:
        """Layer annotation for a single B-scan.

        Args:
            eyevolumelayerannotation: EyeVolumeLayerAnnotation object
            index: Index of the B-scan

        Returns:
            None
        """
        self.eyevolumelayerannotation = eyevolumelayerannotation
        self.volume = eyevolumelayerannotation.volume
        self.index = index

    @property
    def name(self) -> str:
        """Name of the layer annotation."""
        return self.eyevolumelayerannotation.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.eyevolumelayerannotation.meta['name'] = value

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Layer heights."""
        return self.eyevolumelayerannotation.data[self.index, :]

    @data.setter
    def data(self, value: npt.NDArray[np.float64]) -> None:
        self.eyevolumelayerannotation.data[self.index, :] = value

    @property
    def knots(self) -> list:
        """Knots parameterizing the layer heights."""
        return self.eyevolumelayerannotation.knots[self.index]

    @knots.setter
    def knots(self, value: list) -> None:
        self.eyevolumelayerannotation.knots[self.index] = value


class EyeBscanSlabAnnotation:

    def __init__(self, eyevolumeslabannotation: EyeVolumeSlabAnnotation,
                 index: int) -> None:
        """Slab annotation for a single B-scan.

        Args:
            eyevolumeslabannotation: EyeVolumeSlabAnnotation object
            index: Index of the B-scan

        Returns:
            None
        """
        self.eyevolumeslabannotation = eyevolumeslabannotation
        self.volume = eyevolumeslabannotation.volume
        self.index = index

    @property
    def name(self) -> str:
        """Name of the slab annotation."""
        return self.eyevolumeslabannotation.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.eyevolumeslabannotation.meta['name'] = value

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Mask of the slab in the B-scan."""
        return self.eyevolumeslabannotation.mask[self.index, :, :]

    @mask.setter
    def mask(self, value: npt.NDArray[np.bool_]) -> None:
        """Set the mask for the slab in the B-scan."""
        self.eyevolumeslabannotation._mask[self.index, :, :] = value


class EyeEnfacePixelAnnotation:

    def __init__(
        self,
        enface: EyeEnface,
        data: Optional[npt.NDArray[np.bool_]] = None,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Pixel annotation for an enface image.

        Args:
            enface: EyeEnface object
            data: Pixel annotation data
            meta: Metadata
            **kwargs: Additional metadata specified as keyword arguments

        Returns:
            None
        """
        self.enface = enface

        if data is None:
            self.data = np.full(self.enface.shape,
                                fill_value=False,
                                dtype=bool)
        else:
            self.data = data

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        if 'name' not in self.meta:
            self.meta['name'] = 'Pixel Annotation'

    @property
    def name(self) -> str:
        """Name of the pixel annotation."""
        return self.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.meta['name'] = value
