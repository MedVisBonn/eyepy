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
from skimage import draw
from skimage import measure
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


class PolygonAnnotation:
    """Immutable polygon annotation with transformation methods.

    This is a base class for polygon-based annotations like optic disc,
    macula, lesions, etc. The polygon is stored internally and
    transformations return new instances.

    **Coordinate System:** All coordinates are in (row, col) format
    consistent with NumPy array indexing and scikit-image conventions.
    Row corresponds to the vertical axis (y-direction), and col
    corresponds to the horizontal axis (x-direction).
    """

    def __init__(self, polygon: npt.NDArray[np.float64],
                 shape: Optional[tuple[int, int]] = None) -> None:
        """Initialize PolygonAnnotation from polygon vertices.

        Args:
            polygon: Nx2 array of (row, col) coordinates defining the polygon vertices
            shape: Shape (height, width) of the image for mask generation. Optional,
                   can be used later if needed for mask generation.
        """
        self._polygon = np.asarray(polygon, dtype=np.float64)
        if self._polygon.ndim != 2 or self._polygon.shape[1] != 2:
            raise ValueError('Polygon must be an Nx2 array of (row, col) coordinates')

        self._shape = shape
        self._cached_mask = None

    @classmethod
    def from_mask(cls, mask: npt.NDArray[np.bool_]) -> 'PolygonAnnotation':
        """Create PolygonAnnotation from a semantic segmentation mask.

        Args:
            mask: Binary mask where True indicates annotated pixels

        Returns:
            PolygonAnnotation instance with shape set to mask.shape
        """
        # Find contours in the mask
        contours = measure.find_contours(mask.astype(np.float64), level=0.5)

        if len(contours) == 0:
            raise ValueError('No contours found in mask')

        # Take the largest contour (returns row, col)
        largest_contour = max(contours, key=len)

        return cls(polygon=largest_contour, shape=mask.shape)

    @property
    def polygon(self) -> npt.NDArray[np.float64]:
        """Get the polygon representation as Nx2 array of (row, col)
        coordinates."""
        return self._polygon.copy()

    @property
    def shape(self) -> Optional[tuple[int, int]]:
        """Get the image shape used for mask generation."""
        return self._shape

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Generate semantic segmentation mask from polygon.

        Returns:
            Binary mask of shape self.shape

        Raises:
            ValueError: If shape is not set
        """
        if self._shape is None:
            raise ValueError('Shape must be set before generating masdrawk')

        if self._cached_mask is not None:
            return self._cached_mask

        # Create mask from polygon (draw.polygon expects row, col)
        mask = np.zeros(self._shape, dtype=bool)
        rr, cc = draw.polygon(self._polygon[:, 0], self._polygon[:, 1],
                             shape=self._shape)
        mask[rr, cc] = True

        self._cached_mask = mask
        return mask

    def scale(self, factor: float, center: Optional[tuple[float, float]] = None) -> 'PolygonAnnotation':
        """Return a new PolygonAnnotation with scaled polygon.

        Args:
            factor: Scaling factor (e.g., 1.5 for 150% size, 0.5 for 50% size)
            center: Center point (row, col) for scaling. If None, uses the polygon's centroid.

        Returns:
            New PolygonAnnotation instance with scaled polygon

        Example:
            >>> ann_larger = ann.scale(1.5)  # 50% larger
            >>> ann_smaller = ann.scale(0.5)  # 50% smaller
        """
        # Create scaling matrix
        scale_matrix = np.array([
            [factor, 0],
            [0, factor]
        ])

        return self.transform(scale_matrix, center=center)

    def translate(self, drow: float, dcol: float) -> 'PolygonAnnotation':
        """Return a new PolygonAnnotation with translated polygon.

        Args:
            drow: Translation offset in the row direction (vertical). Positive moves downward.
            dcol: Translation offset in the col direction (horizontal). Positive moves rightward.

        Returns:
            New PolygonAnnotation instance with translated polygon

        Example:
            >>> ann_moved = ann.translate(10, 20)  # Move 10 pixels down, 20 pixels right
        """
        # Create translation matrix (2x3 format with translation vector)
        translation_matrix = np.array([
            [1, 0, drow],
            [0, 1, dcol]
        ])

        return self.transform(translation_matrix)

    def rotate(self, angle: float, center: Optional[tuple[float, float]] = None) -> 'PolygonAnnotation':
        """Return a new PolygonAnnotation with rotated polygon.

        Args:
            angle: Rotation angle in radians (positive = counter-clockwise)
            center: Center point (row, col) for rotation. If None, uses the polygon's centroid.

        Returns:
            New PolygonAnnotation instance with rotated polygon

        Example:
            >>> ann_rotated = ann.rotate(np.pi / 4)  # Rotate 45 degrees counter-clockwise
        """
        # Create rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        return self.transform(rotation_matrix, center=center)

    def transform(self, matrix: npt.NDArray[np.float64],
                  center: Optional[tuple[float, float]] = None) -> 'PolygonAnnotation':
        """Return a new PolygonAnnotation with affine-transformed polygon.

        Args:
            matrix: 2x2 affine transformation matrix or 2x3 matrix (last column is translation)
            center: Center point (row, col) for transformation. If None, uses the polygon's centroid.
                   Only used if matrix is 2x2 (ignored for 2x3 matrices with translation).

        Returns:
            New PolygonAnnotation instance with transformed polygon

        Example:
            >>> # Shear transformation
            >>> shear_matrix = np.array([[1, 0.5], [0, 1]])
            >>> ann_sheared = ann.transform(shear_matrix)
            >>>
            >>> # Combined scale and translate
            >>> matrix = np.array([[1.5, 0, 10], [0, 1.5, 20]])  # scale 1.5x, translate (10, 20)
            >>> ann_transformed = ann.transform(matrix)
        """
        if matrix.shape == (2, 3):
            # Affine transformation with translation
            # Extract rotation/scale part and translation
            transform_matrix = matrix[:, :2]
            translation = matrix[:, 2]

            # polygon is (row, col), matrix also expects (row, col)
            transformed = self._polygon @ transform_matrix.T + translation
            result = transformed

        elif matrix.shape == (2, 2):
            # Pure linear transformation (rotation, scale, shear)
            if center is None:
                center = self._polygon.mean(axis=0)

            # Translate to origin
            centered = self._polygon - np.array(center)

            # polygon is (row, col), matrix also expects (row, col)
            transformed = centered @ matrix.T

            # Translate back
            result = transformed + np.array(center)
        else:
            raise ValueError('Matrix must be 2x2 or 2x3')

        return self.__class__(result, shape=self._shape)

    def plot(self, ax: Optional[plt.Axes] = None, offset: tuple[float, float] = (0, 0),
             **kwargs) -> None:
        """Plot the polygon outline on the given axes.

        Args:
            ax: Matplotlib axes. If None, uses current axes (plt.gca())
            offset: (row_offset, col_offset) to adjust polygon position before plotting
            **kwargs: Additional keyword arguments passed to ax.plot()

        Returns:
            None
        """
        if ax is None:
            ax = plt.gca()

        # Apply offset to polygon coordinates
        row_offset, col_offset = offset
        polygon = self._polygon.copy()
        polygon[:, 0] -= row_offset
        polygon[:, 1] -= col_offset

        # Plot polygon outline (matplotlib expects x, y, so swap col, row)
        ax.plot(polygon[:, 1], polygon[:, 0], **kwargs)
        # Close the polygon
        ax.plot([polygon[-1, 1], polygon[0, 1]],
               [polygon[-1, 0], polygon[0, 0]], **kwargs)


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

        # Set default color from config if not already specified
        if 'color' not in self.meta:
            self.meta['color'] = '#' + config.area_colors[self.meta['name']]

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

        # Set default color from config if not already specified
        if 'color' not in self.meta:
            self.meta['color'] = '#' + config.area_colors[self.meta['name']]

    @property
    def name(self) -> str:
        """Name of the pixel annotation."""
        return self.meta['name']

    @name.setter
    def name(self, value: str) -> None:
        self.meta['name'] = value


class EyeEnfaceOpticDiscAnnotation(PolygonAnnotation):
    """Optic disc annotation for enface images with ellipse fitting capabilities.

    Inherits from PolygonAnnotation and adds optic disc-specific features:
    - Creation from ellipse parameters
    - Fitted ellipse properties (center, width, height)

    The optic disc can be initialized from an ellipse, a polygon, or a semantic mask.
    Internally, it is stored as a polygon, but provides properties to access ellipse
    parameters fitted to the polygon.
    """

    def __init__(self, polygon: npt.NDArray[np.float64],
                 shape: Optional[tuple[int, int]] = None) -> None:
        """Initialize EyeEnfaceOpticDiscAnnotation from a polygon.

        Args:
            polygon: Nx2 array of (row, col) coordinates defining the polygon vertices
            shape: Shape (height, width) of the image for mask generation. Optional,
                   can be used later if needed for mask generation.
        """
        super().__init__(polygon, shape)
        self._cached_ellipse = None

    @classmethod
    def from_ellipse(cls, center: tuple[float, float], minor_axis: float, major_axis: float,
                     rotation: float = 0.0, num_points: int = 64,
                     shape: Optional[tuple[int, int]] = None) -> 'EyeEnfaceOpticDiscAnnotation':
        """Create EyeEnfaceOpticDiscAnnotation from ellipse parameters.

        Args:
            center: (row, col) coordinates of ellipse center
            minor_axis: Length of the minor axis (shorter diameter)
            major_axis: Length of the major axis (longer diameter)
            rotation: Rotation angle in radians (rotates the ellipse axes)
            num_points: Number of points to sample on the ellipse perimeter
            shape: Shape (height, width) of the image for mask generation

        Returns:
            EyeEnfaceOpticDiscAnnotation instance

        Note:
            Before rotation, the minor axis is aligned with the row direction (vertical)
            and the major axis is aligned with the col direction (horizontal).
        """
        # Generate points on ellipse perimeter
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Ellipse in standard position (row, col offsets)
        # Minor axis along row direction, major axis along col direction
        row_offset = (minor_axis / 2) * np.cos(theta)
        col_offset = (major_axis / 2) * np.sin(theta)

        # Apply rotation
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        row_rot = row_offset * cos_rot - col_offset * sin_rot
        col_rot = row_offset * sin_rot + col_offset * cos_rot

        # Translate to center
        row_coords = row_rot + center[0]
        col_coords = col_rot + center[1]

        polygon = np.column_stack([row_coords, col_coords])
        return cls(polygon=polygon, shape=shape)

    def _fit_ellipse(self) -> tuple[float, float, float, float, float]:
        """Fit an ellipse to the polygon.

        Returns:
            Tuple of (center_row, center_col, width, height, rotation)
        """
        if self._cached_ellipse is not None:
            return self._cached_ellipse

        # Always estimate shape from polygon bounds to handle negative coordinates
        min_row, min_col = self._polygon.min(axis=0)
        max_row, max_col = self._polygon.max(axis=0)

        # Add padding
        padding = 5
        height = int(np.ceil(max_row - min_row)) + 2 * padding
        width = int(np.ceil(max_col - min_col)) + 2 * padding
        temp_shape = (height, width)

        # Adjust polygon coordinates to non-negative with padding
        adjusted_polygon = self._polygon - [min_row - padding, min_col - padding]

        # Create temporary mask (draw.polygon expects row, col)
        temp_mask = np.zeros(temp_shape, dtype=bool)
        rr, cc = draw.polygon(adjusted_polygon[:, 0], adjusted_polygon[:, 1],
                             shape=temp_shape)
        temp_mask[rr, cc] = True

        # Get region properties
        regions = measure.regionprops(temp_mask.astype(np.uint8))
        if len(regions) == 0:
            raise ValueError('Could not fit ellipse to polygon')

        region = regions[0]

        # Get ellipse parameters from region properties (centroid is row, col)
        row0, col0 = region.centroid

        # Get orientation and axis lengths
        orientation = region.orientation

        # The major and minor axis lengths
        minor_axis_length = region.axis_minor_length
        major_axis_length = region.axis_major_length

        # Adjust center back to original coordinates
        row0 += (min_row - padding)
        col0 += (min_col - padding)

        self._cached_ellipse = (row0, col0, minor_axis_length, major_axis_length, orientation)
        return self._cached_ellipse

    @property
    def center(self) -> tuple[float, float]:
        """Get the center (row, col) of the fitted ellipse.

        Returns:
            Tuple of (center_row, center_col)
        """
        row0, col0, _, _, _ = self._fit_ellipse()
        return (row0, col0)

    @property
    def width(self) -> float:
        """Get the horizontal extent of the optic disc polygon.

        This measures the horizontal (column-direction) size by finding the
        horizontal line passing through the center that intersects the polygon.

        Returns:
            Horizontal extent (width) of the polygon through its center
        """
        center = self.center
        center_row = center[0]

        # Find min and max column coordinates where the polygon exists at this row level
        # We'll check all edges of the polygon for intersections with horizontal line
        polygon = self._polygon
        col_coords = []

        for i in range(len(polygon)):
            row1, col1 = polygon[i]
            row2, col2 = polygon[(i + 1) % len(polygon)]

            # Check if the edge crosses the horizontal line at center_row
            if min(row1, row2) <= center_row <= max(row1, row2):
                if abs(row2 - row1) < 1e-10:
                    # Horizontal edge at center_row
                    col_coords.extend([col1, col2])
                else:
                    # Interpolate to find col at center_row
                    t = (center_row - row1) / (row2 - row1)
                    col_intersect = col1 + t * (col2 - col1)
                    col_coords.append(col_intersect)

        if len(col_coords) < 2:
            # Fallback: use bounding box width
            return self._polygon[:, 1].max() - self._polygon[:, 1].min()

        # Width is the distance between leftmost and rightmost intersections
        return max(col_coords) - min(col_coords)

    @property
    def height(self) -> float:
        """Get the vertical extent of the optic disc polygon.

        This measures the vertical (row-direction) size by finding the
        vertical line passing through the center that intersects the polygon.

        Returns:
            Vertical extent (height) of the polygon through its center
        """
        center = self.center
        center_col = center[1]

        # Find min and max row coordinates where the polygon exists at this col level
        polygon = self._polygon
        row_coords = []

        for i in range(len(polygon)):
            row1, col1 = polygon[i]
            row2, col2 = polygon[(i + 1) % len(polygon)]

            # Check if the edge crosses the vertical line at center_col
            if min(col1, col2) <= center_col <= max(col1, col2):
                if abs(col2 - col1) < 1e-10:
                    # Vertical edge at center_col
                    row_coords.extend([row1, row2])
                else:
                    # Interpolate to find row at center_col
                    t = (center_col - col1) / (col2 - col1)
                    row_intersect = row1 + t * (row2 - row1)
                    row_coords.append(row_intersect)

        if len(row_coords) < 2:
            # Fallback: use bounding box height
            return self._polygon[:, 0].max() - self._polygon[:, 0].min()

        # Height is the distance between topmost and bottommost intersections
        return max(row_coords) - min(row_coords)

    def plot(self, ax: Optional[plt.Axes] = None, offset: tuple[float, float] = (0, 0),
             plot_contour: bool = True, plot_area: bool = False,
             contour_color: str = 'red', contour_linewidth: float = 2,
             contour_linestyle: str = '-', area_color: Optional[str] = None,
             area_alpha: float = 0.3, **kwargs) -> None:
        """Plot the optic disc annotation on the given axes.

        Provides flexible visualization options including contour outline and/or filled area.

        Args:
            ax: Matplotlib axes to plot on. If None, uses current axes (plt.gca())
            offset: (row_offset, col_offset) to apply to polygon coordinates for region plotting
            plot_contour: If True, plot the contour outline (default: True)
            plot_area: If True, plot the filled area (default: False)
            contour_color: Color of the contour outline (default: 'red')
            contour_linewidth: Line width of the contour outline (default: 2)
            contour_linestyle: Line style of the contour outline (default: '-')
            area_color: Color of the filled area. If None, uses contour_color (default: None)
            area_alpha: Alpha transparency of the filled area (default: 0.3)
            **kwargs: Additional keyword arguments. For contour-only plotting, passed to ax.plot().
                     For area plotting, can include 'edgecolor' and 'facecolor' for fine control.

        Returns:
            None

        Example:
            >>> # Plot only contour (default)
            >>> optic_disc.plot(ax, contour_color='red', contour_linewidth=2)
            >>>
            >>> # Plot only filled area
            >>> optic_disc.plot(ax, plot_contour=False, plot_area=True, area_color='red', area_alpha=0.5)
            >>>
            >>> # Plot both contour and area
            >>> optic_disc.plot(ax, plot_contour=True, plot_area=True,
            ...                 contour_color='darkred', area_color='red', area_alpha=0.3)
        """
        if ax is None:
            ax = plt.gca()

        # Apply offset to polygon coordinates
        row_offset, col_offset = offset
        polygon = self._polygon.copy()
        polygon[:, 0] -= row_offset
        polygon[:, 1] -= col_offset

        # Determine area color if not specified
        if area_color is None:
            area_color = contour_color

        # Plot filled area if requested
        if plot_area:
            # Extract face/edge color from kwargs if provided, otherwise use our parameters
            facecolor = kwargs.pop('facecolor', area_color)
            edgecolor = kwargs.pop('edgecolor', contour_color if plot_contour else 'none')

            # Use matplotlib's fill to create filled polygon
            # polygon is (row, col), matplotlib expects (x, y) = (col, row)
            ax.fill(polygon[:, 1], polygon[:, 0],
                   facecolor=facecolor,
                   edgecolor=edgecolor if plot_contour else 'none',
                   alpha=area_alpha,
                   linewidth=contour_linewidth if plot_contour else 0,
                   linestyle=contour_linestyle if plot_contour else '-')

        # Plot contour if requested (and not already plotted as part of area)
        elif plot_contour:
            # Combine contour styling
            contour_kwargs = {
                'color': contour_color,
                'linewidth': contour_linewidth,
                'linestyle': contour_linestyle,
                **kwargs
            }

            # Plot polygon outline (polygon is x, y, matplotlib expects x, y)
            ax.plot(polygon[:, 1], polygon[:, 0], **contour_kwargs)
            # Close the polygon
            ax.plot([polygon[-1, 1], polygon[0, 1]],
                   [polygon[-1, 0], polygon[0, 0]], **contour_kwargs)


class EyeEnfaceFoveaAnnotation(PolygonAnnotation):
    """Fovea annotation for enface images with center point detection.

    Inherits from PolygonAnnotation and adds fovea-specific features:
    - Simple center point calculation
    - Can be created from a small circular region

    The fovea is typically a small region, so the center is calculated
    as the mean of the polygon vertices.
    """

    @property
    def center(self) -> tuple[float, float]:
        """Get the center (row, col) of the fovea.

        Returns:
            Tuple of (center_row, center_col) calculated as the mean of polygon vertices
        """
        return tuple(self._polygon.mean(axis=0))

    def plot(self, ax: Optional[plt.Axes] = None, offset: tuple[float, float] = (0, 0),
             color: str = 'yellow', marker: str = '+', markersize: float = 12,
             markeredgewidth: float = 2, **kwargs) -> None:
        """Plot the fovea annotation on the given axes as a center marker.

        Args:
            ax: Matplotlib axes to plot on. If None, uses current axes (plt.gca())
            offset: (row_offset, col_offset) to apply to coordinates for region plotting
            color: Color of the fovea marker (default: 'yellow')
            marker: Marker style (default: '+')
            markersize: Size of the marker (default: 12)
            markeredgewidth: Width of the marker edge (default: 2)
            **kwargs: Additional keyword arguments passed to ax.plot() for styling

        Returns:
            None
        """
        if ax is None:
            ax = plt.gca()

        # Combine default styling with user overrides
        plot_kwargs = {
            'color': color,
            'marker': marker,
            'markersize': markersize,
            'markeredgewidth': markeredgewidth,
            **kwargs
        }

        # Apply offset to center coordinates
        row_offset, col_offset = offset
        center = self.center
        center_row = center[0] - row_offset
        center_col = center[1] - col_offset

        ax.plot(center_col, center_row, **plot_kwargs)
