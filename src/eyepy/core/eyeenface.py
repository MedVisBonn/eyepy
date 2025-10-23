from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np
from scipy import ndimage
from skimage import transform

from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
from eyepy.core.annotations import EyeEnfacePixelAnnotation
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.plotting import plot_scalebar
from eyepy.core.plotting import plot_watermark

if TYPE_CHECKING:
    pass


class EyeEnface:
    """Enface (2D projection) image with optional anatomical annotations.

    **Coordinate System Convention:**
    EyeEnface uses (row, col) image pixel coordinates:
    - row: vertical axis, increases downward (corresponds to y)
    - col: horizontal axis, increases rightward (corresponds to x)

    All image operations (scaling, translation, rotation) work in this space.
    Physical coordinates (x, y) in millimeters are converted to (col, row)
    format when needed for internal calculations.

    Annotations (optic disc, fovea, pixel annotations) are stored and
    transformed in (row, col) coordinates.
    """

    def __init__(self, data: npt.NDArray[np.int64],
                 meta: EyeEnfaceMeta,
                 optic_disc: Optional[EyeEnfaceOpticDiscAnnotation] = None,
                 fovea: Optional[EyeEnfaceFoveaAnnotation] = None) -> None:
        """Initialize EyeEnface.

        Args:
            data: Enface image data
            meta: Metadata for the enface image
            optic_disc: Optional EyeEnfaceOpticDiscAnnotation
            fovea: Optional EyeEnfaceFoveaAnnotation
        """
        self.data = data
        self._area_maps = []
        self.meta = meta
        self._optic_disc = optic_disc
        self._fovea = fovea

        # Validate and set laterality if both optic disc and fovea are provided
        if optic_disc is not None and fovea is not None:
            self._infer_and_validate_laterality()

    @property
    def optic_disc(self) -> Optional[EyeEnfaceOpticDiscAnnotation]:
        """Get the optic disc annotation.

        Returns:
            The optic disc annotation or None if not set
        """
        return self._optic_disc

    @optic_disc.setter
    def optic_disc(self, value: Optional[EyeEnfaceOpticDiscAnnotation]) -> None:
        """Set the optic disc annotation.

        If both optic_disc and fovea are set, laterality will be inferred (if not already set)
        and validated.

        Args:
            value: The optic disc annotation or None

        Raises:
            ValueError: If laterality validation fails
        """
        self._optic_disc = value
        # Validate and infer laterality if both annotations are now present
        if self._optic_disc is not None and self._fovea is not None:
            self._infer_and_validate_laterality()

    @property
    def fovea(self) -> Optional[EyeEnfaceFoveaAnnotation]:
        """Get the fovea annotation.

        Returns:
            The fovea annotation or None if not set
        """
        return self._fovea

    @fovea.setter
    def fovea(self, value: Optional[EyeEnfaceFoveaAnnotation]) -> None:
        """Set the fovea annotation.

        If both optic_disc and fovea are set, laterality will be inferred (if not already set)
        and validated.

        Args:
            value: The fovea annotation or None

        Raises:
            ValueError: If laterality validation fails
        """
        self._fovea = value
        # Validate and infer laterality if both annotations are now present
        if self._optic_disc is not None and self._fovea is not None:
            self._infer_and_validate_laterality()

    def _infer_and_validate_laterality(self) -> None:
        """Infer laterality if not set, then validate it.

        If laterality is not set in metadata, infer it from the relative positions
        of optic disc and fovea and set it in the metadata. Then validate that the
        laterality matches the anatomical positions.

        Raises:
            ValueError: If laterality validation fails
        """
        if self._optic_disc is None or self._fovea is None:
            return

        laterality = self.meta.get('laterality', None)

        # If laterality is not set, infer and set it
        if laterality is None:
            inferred = self._infer_laterality()
            if inferred is not None:
                self.meta['laterality'] = inferred
                return

        # Now validate the laterality
        self._validate_laterality()

    def _infer_laterality(self) -> Optional[str]:
        """Infer laterality from the relative positions of optic disc and fovea.

        For anatomically correct images:
        - If optic disc is to the right of fovea -> Right eye (OD)
        - If optic disc is to the left of fovea -> Left eye (OS)

        Returns:
            'OD' for right eye, 'OS' for left eye, or None if cannot infer
        """
        if self._optic_disc is None or self._fovea is None:
            return None

        od_center = self._optic_disc.center
        fovea_center = self._fovea.center

        # Compare col-coordinates (center returns (row, col) format)
        od_col = od_center[1]
        fovea_col = fovea_center[1]

        # If optic disc is to the right of fovea -> right eye
        if od_col > fovea_col:
            return 'OD'
        # If optic disc is to the left of fovea -> left eye
        elif od_col < fovea_col:
            return 'OS'
        else:
            # They are at the same column position, cannot infer
            return None

    def _validate_laterality(self) -> None:
        """Validate that laterality matches the relative position of optic disc
        and fovea.

        For a right eye (OD), the optic disc should be to the right of the fovea (higher col).
        For a left eye (OS), the optic disc should be to the left of the fovea (lower col).

        Raises:
            ValueError: If the laterality doesn't match the anatomical positions
        """
        if self._optic_disc is None or self._fovea is None:
            return

        laterality = self.meta.get('laterality', None)
        if laterality is None:
            # No laterality info, skip validation
            return

        # Get centers
        od_center = self._optic_disc.center
        fovea_center = self._fovea.center

        # Compare col-coordinates (center returns (row, col) format)
        od_col = od_center[1]
        fovea_col = fovea_center[1]

        # For right eye (OD): optic disc should be to the right of fovea
        # For left eye (OS): optic disc should be to the left of fovea
        if laterality.upper() in ['OD', 'R', 'RIGHT']:
            if od_col <= fovea_col:
                raise ValueError(
                    f'Laterality mismatch: Right eye (OD) expects optic disc to the right of fovea, '
                    f'but optic disc is at col={od_col:.1f} and fovea is at col={fovea_col:.1f}'
                )
        elif laterality.upper() in ['OS', 'L', 'LEFT']:
            if od_col >= fovea_col:
                raise ValueError(
                    f'Laterality mismatch: Left eye (OS) expects optic disc to the left of fovea, '
                    f'but optic disc is at col={od_col:.1f} and fovea is at col={fovea_col:.1f}'
                )

    @property
    def area_maps(self) -> dict[str, EyeEnfacePixelAnnotation]:
        """

        Returns:

        """
        # Create a dict to access area_maps by their name
        return {am.name: am for am in self._area_maps}

    def add_area_annotation(self,
                            area_map: Optional[npt.NDArray[np.bool_]] = None,
                            meta: Optional[dict] = None,
                            **kwargs: Any) -> EyeEnfacePixelAnnotation:
        """

        Args:
            area_map:
            meta:
            **kwargs:

        Returns:

        """
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        area_annotation = EyeEnfacePixelAnnotation(self, area_map, meta)
        self._area_maps.append(area_annotation)
        return area_annotation

    @property
    def scale_x(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_x']

    @scale_x.setter
    def scale_x(self, value: float) -> None:
        self.meta['scale_x'] = value

    @property
    def scale_y(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_y']

    @scale_y.setter
    def scale_y(self, value: float) -> None:
        self.meta['scale_y'] = value

    @property
    def scale_unit(self) -> str:
        """

        Returns:

        """
        return self.meta['scale_unit']

    @scale_unit.setter
    def scale_unit(self, value: str) -> None:
        self.meta['scale_unit'] = value

    @property
    def size_x(self) -> int:
        """

        Returns:

        """
        return self.shape[1]

    @property
    def size_y(self) -> int:
        """

        Returns:

        """
        return self.shape[0]

    @property
    def laterality(self) -> str:
        """

        Returns:

        """
        return self.meta['laterality']

    @property
    def shape(self) -> tuple[int, int]:
        """

        Returns:

        """
        return self.data.shape

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        region: tuple[slice, slice] = np.s_[:, :],
        scalebar: Union[bool, str] = 'botleft',
        scalebar_kwargs: Optional[dict[str, Any]] = None,
        watermark: bool = True,
        areas: Union[bool, list[str]] = True,
        plot_optic_disc: bool = True,
        plot_fovea: bool = True,
        area_kwargs: Optional[dict[str, Any]] = None,
        optic_disc_kwargs: Optional[dict[str, Any]] = None,
        fovea_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """

        Args:
            ax: Axes to plot on. If not provided plot on the current axes (plt.gca()).
            region: Region of the localizer to plot (default: `np.s_[:, :]`)
            scalebar: Position of the scalebar, one of "topright", "topleft", "botright", "botleft" or `False` (default: "botleft"). If `True` the scalebar is placed in the bottom left corner. You can custumize the scalebar using the `scalebar_kwargs` argument.
            scalebar_kwargs: Optional keyword arguments for customizing the scalebar. Check the documentation of [plot_scalebar][eyepy.core.plotting.plot_scalebar] for more information.
            watermark: If `True` plot a watermark on the image (default: `True`). When removing the watermark, please consider to cite eyepy in your publication.
            areas: If `True` plot all area annotations (default: `True`). If a list of strings is given, plot the area annotations with the given names.
            plot_optic_disc: If `True` and optic_disc is available, plot the optic disc annotation (default: `True`).
            plot_fovea: If `True` and fovea is available, plot the fovea annotation (default: `True`).
            area_kwargs: Optional keyword arguments for customizing the area annotations (alpha, etc.).
            optic_disc_kwargs: Optional keyword arguments for customizing the optic disc plot (color, linewidth, etc.).
            fovea_kwargs: Optional keyword arguments for customizing the fovea plot (color, marker, markersize, etc.).
        Returns:
            None

        """
        ax = plt.gca() if ax is None else ax
        vmin = np.min(self.data)
        vmax = np.max(self.data)

        ax.imshow(self.data[region], cmap='gray', vmin=vmin, vmax=vmax)

        # Make sure tick labels match the image region
        y_start = region[0].start if region[0].start is not None else 0
        x_start = region[1].start if region[1].start is not None else 0
        y_end = region[0].stop if region[0].stop is not None else self.size_y
        x_end = region[1].stop if region[1].stop is not None else self.size_x

        # Ticks are not clipped to the image region. Clip them here.
        yticks = ax.get_yticks()
        yticks = yticks[np.nonzero(
            np.logical_and(yticks >= 0, yticks <= y_end - y_start - 1))]
        xticks = ax.get_xticks()
        xticks = xticks[np.nonzero(
            np.logical_and(xticks >= 0, xticks <= x_end - x_start - 1))]

        # Set clipped ticks (this is only necessary because we change the labels later)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)

        # Set labels to ticks + start of the region as an offset
        ax.set_yticklabels([str(int(t + y_start)) for t in yticks])
        ax.set_xticklabels([str(int(t + x_start)) for t in xticks])

        if scalebar:
            if scalebar_kwargs is None:
                scalebar_kwargs = {}

            scale_unit = self.meta['scale_unit']
            scalebar_kwargs = {
                **{
                    'scale': (self.scale_x, self.scale_y),
                    'scale_unit': scale_unit
                },
                **scalebar_kwargs
            }

            if not 'pos' in scalebar_kwargs:
                sx = x_end - x_start
                sy = y_end - y_start

                if scalebar is True:
                    scalebar = 'botleft'

                if scalebar == 'botleft':
                    scalebar_kwargs['pos'] = (sx - 0.95 * sx, 0.95 * sy)
                elif scalebar == 'botright':
                    scalebar_kwargs['pos'] = (0.95 * sx, 0.95 * sy)
                    scalebar_kwargs['flip_x'] = True
                elif scalebar == 'topleft':
                    scalebar_kwargs['pos'] = (sx - 0.95 * sx, 0.05 * sy)
                    scalebar_kwargs['flip_y'] = True
                elif scalebar == 'topright':
                    scalebar_kwargs['pos'] = (0.95 * sx, 0.05 * sy)
                    scalebar_kwargs['flip_x'] = True
                    scalebar_kwargs['flip_y'] = True

            plot_scalebar(ax=ax, **scalebar_kwargs)

        if watermark:
            plot_watermark(ax)

        # Handle areas parameter
        if not areas:
            areas_to_plot = []
        elif areas is True:
            areas_to_plot = list(self.area_maps.keys())
        else:
            areas_to_plot = areas

        # Set default area_kwargs if not provided
        if area_kwargs is None:
            area_kwargs = {'alpha': 0.5}

        # Plot area annotations
        for area_name in areas_to_plot:
            if area_name not in self.area_maps:
                continue

            area_annotation = self.area_maps[area_name]
            data = area_annotation.data[region]
            visible = np.zeros(data.shape, dtype=bool)
            visible[data != 0] = 1.0

            # Get color from metadata or use default
            meta = area_annotation.meta
            color = meta.get('color', 'red')
            color = mcolors.to_rgba(color)

            # Create a 0 radius circle patch as dummy for the area label
            patch = mpatches.Circle((0, 0), radius=0, color=color, label=area_name)
            ax.add_patch(patch)

            # Create plot_data by tiling the color vector over the plotting shape
            plot_data = np.tile(np.array(color), data.shape + (1, ))
            # Now turn the alpha channel 0 where the mask is 0 and adjust the remaining alpha
            plot_data[..., 3] *= visible * area_kwargs.get('alpha', 0.5)

            ax.imshow(
                plot_data,
                interpolation='none',
            )

        # Plot optic disc if available
        if plot_optic_disc and self.optic_disc is not None:
            if optic_disc_kwargs is None:
                optic_disc_kwargs = {}

            # Calculate region offset
            y_start = region[0].start if region[0].start is not None else 0
            x_start = region[1].start if region[1].start is not None else 0

            # Delegate to the optic disc's plot method
            self.optic_disc.plot(ax=ax, offset=(x_start, y_start), **optic_disc_kwargs)

        # Plot fovea if available
        if plot_fovea and self.fovea is not None:
            if fovea_kwargs is None:
                fovea_kwargs = {}

            # Calculate region offset
            y_start = region[0].start if region[0].start is not None else 0
            x_start = region[1].start if region[1].start is not None else 0

            # Delegate to the fovea's plot method
            self.fovea.plot(ax=ax, offset=(x_start, y_start), **fovea_kwargs)

    def scale(self, scale_y: float, scale_x: float,
              order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Scale the enface image and all annotations.

        Returns a new EyeEnface instance with scaled image data and transformed annotations.
        The original EyeEnface remains unchanged.

        Args:
            scale_y: Scaling factor for the y-axis
            scale_x: Scaling factor for the x-axis
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with scaled data and annotations
        """
        # Create affine transformation matrix for scaling
        matrix = np.array([
            [scale_y, 0, 0],
            [0, scale_x, 0],
            [0, 0, 1]
        ])

        # Calculate output shape
        output_shape = (int(self.shape[0] * scale_y), int(self.shape[1] * scale_x))

        return self.transform(matrix, output_shape=output_shape, order=order, mode=mode, cval=cval)

    def translate(self, drow: float, dcol: float,
                  order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Translate the enface image and all annotations.

        Returns a new EyeEnface instance with translated image data and transformed annotations.
        The original EyeEnface remains unchanged.

        Args:
            drow: Translation offset in the row direction (vertical, pixels)
            dcol: Translation offset in the col direction (horizontal, pixels)
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with translated data and annotations
        """
        # Create affine transformation matrix for translation
        # Matrix format: [[1, 0, drow], [0, 1, dcol], [0, 0, 1]]
        matrix = np.array([
            [1, 0, drow],
            [0, 1, dcol],
            [0, 0, 1]
        ])

        return self.transform(matrix, order=order, mode=mode, cval=cval)

    def rotate(self, angle: float, center: Optional[tuple[float, float]] = None,
               order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Rotate the enface image and all annotations.

        Returns a new EyeEnface instance with rotated image data and transformed annotations.
        The original EyeEnface remains unchanged.

        Args:
            angle: Rotation angle in degrees (positive = counter-clockwise)
            center: Center of rotation (row, col). If None, uses image center
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with rotated data and annotations
        """
        if center is None:
            center = (self.shape[0] / 2, self.shape[1] / 2)

        # Build rotation matrix in (row, col) coordinates for counter-clockwise rotation
        # Standard mathematical rotation: positive angle = counter-clockwise
        # The standard counter-clockwise rotation matrix is:
        # [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        # But with (row, col) format where row increases downward (y-axis flipped),
        # we need: [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]] to get counter-clockwise rotation
        center_row, center_col = center
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix for (row, col) format with counter-clockwise rotation around (center_row, center_col)
        matrix_rowcol = np.array([
            [cos_a, -sin_a, center_row - cos_a * center_row + sin_a * center_col],
            [sin_a, cos_a, center_col - sin_a * center_row - cos_a * center_col],
            [0, 0, 1]
        ])

        return self.transform(matrix_rowcol, order=order, mode=mode, cval=cval)

    def standard_rotation(self, order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Rotate the enface to align optic disc and fovea centers horizontally.

        This method rotates the image so that the optic disc and fovea centers lie on the
        same horizontal line. The rotation is performed around the midpoint between the OD
        and fovea, preserving their relative positions. The rotation direction depends on
        the laterality:
        - For right eyes (OD): optic disc on the right, fovea on the left
        - For left eyes (OS): optic disc on the left, fovea on the right

        This creates a standardized orientation useful for comparison across different images.

        Requires both optic_disc and fovea annotations to be present.

        Args:
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with standardized rotation

        Raises:
            ValueError: If optic_disc or fovea annotations are missing
        """
        if self._optic_disc is None:
            raise ValueError('standard_rotation requires optic_disc annotation')
        if self._fovea is None:
            raise ValueError('standard_rotation requires fovea annotation')

        # Get centers of optic disc and fovea
        od_center = self.optic_disc.center
        fovea_center = self.fovea.center

        # Determine laterality from metadata or infer from positions
        laterality = self.meta.get('laterality', None)

        # Calculate vector from OD to fovea (in row, col coordinates)
        # center returns (row, col) tuples
        d_row = fovea_center[0] - od_center[0]
        d_col = fovea_center[1] - od_center[1]

        # Calculate the current angle of the OD-to-fovea line
        # In (row, col) space, horizontal line has d_row=0, so angle from horizontal is arctan2(d_row, d_col)
        # Positive angle means counter-clockwise from positive column direction (pointing right)
        current_angle_rad = np.arctan2(d_row, d_col)

        # Determine target angle based on laterality (in row, col coordinates)
        # For OD (right eye): OD on right, fovea on left -> line points left (180°)
        # For OS (left eye): OD on left, fovea on right -> line points right (0°)
        if laterality is not None:
            laterality_upper = str(laterality).upper()
            if laterality_upper in ['OD', 'R', 'RIGHT']:
                target_angle_rad = np.pi  # Pointing left
            elif laterality_upper in ['OS', 'L', 'LEFT']:
                target_angle_rad = 0.0  # Pointing right
            else:
                # Unknown laterality, default to horizontal alignment with fovea to the right
                target_angle_rad = 0.0
        else:
            # No laterality info, align horizontally with fovea to the right
            target_angle_rad = 0.0

        # Calculate rotation angle needed
        rotation_angle_rad = target_angle_rad - current_angle_rad

        # Normalize to [-π, π] to take the shortest rotation path
        rotation_angle_rad = np.arctan2(np.sin(rotation_angle_rad), np.cos(rotation_angle_rad))
        rotation_angle_deg = np.rad2deg(rotation_angle_rad)

        # Rotate around the midpoint between OD and fovea to keep them centered
        # This preserves their relative position better than rotating around image center
        midpoint_row = (od_center[0] + fovea_center[0]) / 2
        midpoint_col = (od_center[1] + fovea_center[1]) / 2
        center = (midpoint_row, midpoint_col)

        return self.rotate(-rotation_angle_deg, center=center, order=order, mode=mode, cval=cval)

    def transform(self, matrix: npt.NDArray[np.float64],
                  output_shape: Optional[tuple[int, int]] = None,
                  order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Apply an affine transformation to the enface image and all annotations.

        Returns a new EyeEnface instance with transformed image data and annotations.
        The original EyeEnface remains unchanged. The metadata is updated to reflect
        any scaling in the transformation.

        Args:
            matrix: 3x3 affine transformation matrix in homogeneous coordinates or 2x3 matrix
            output_shape: Shape of the output image (height, width). If None, uses input shape
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with transformed data and annotations
        """
        if output_shape is None:
            output_shape = self.shape

        # Convert 3x3 to 2x3 for annotations if needed
        if matrix.shape == (3, 3):
            # Matrix is in (row, col) format - use directly for annotations
            annotation_matrix = matrix[:2, :]
        elif matrix.shape == (2, 3):
            annotation_matrix = matrix
        else:
            raise ValueError('Matrix must be 3x3 or 2x3')

        # For image transformation with skimage, we need to convert to (x, y) format
        # skimage's AffineTransform works with (x, y) format, not (row, col)
        # So we need to swap the matrix dimensions: swap rows and columns
        if matrix.shape == (2, 3):
            # Convert 2x3 to 3x3
            full_matrix = np.vstack([matrix, [0, 0, 1]])
        else:
            full_matrix = matrix

        # Convert from (row, col) to (x, y) by swapping the matrix rows/cols
        # If M transforms (row, col) -> (row', col'), then for (x, y) = (col, row):
        # [row']   [a  b  tx]   [row]       [col']   [b  a  ty]   [col]
        # [col'] = [c  d  ty] * [col]  =>   [row'] = [d  c  tx] * [row]
        skimage_matrix = np.array([
            [full_matrix[1, 1], full_matrix[1, 0], full_matrix[1, 2]],
            [full_matrix[0, 1], full_matrix[0, 0], full_matrix[0, 2]],
            [0, 0, 1]
        ])

        # Extract scaling factors from the transformation matrix
        # The scaling factors are the norms of the first two columns
        scale_y = np.linalg.norm(annotation_matrix[:, 0])
        scale_x = np.linalg.norm(annotation_matrix[:, 1])

        # Create updated metadata with adjusted scale_x and scale_y
        # When the image is scaled up, each pixel represents less physical distance
        # Only update metadata if it's a real EyeEnfaceMeta object, not a mock
        if isinstance(self.meta, EyeEnfaceMeta):
            # Real EyeEnfaceMeta object - use the copy() method
            updated_meta = self.meta.copy()
            if 'scale_x' in updated_meta:
                updated_meta['scale_x'] = updated_meta['scale_x'] / scale_x
            if 'scale_y' in updated_meta:
                updated_meta['scale_y'] = updated_meta['scale_y'] / scale_y
        else:
            # Mock or other type - just reuse it
            updated_meta = self.meta

        # Create AffineTransform from matrix
        # Note: scikit-image uses inverse matrix convention
        tform = transform.AffineTransform(matrix=skimage_matrix)

        # Transform the image data
        transformed_data = transform.warp(
            self.data,
            tform.inverse,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
            preserve_range=True
        ).astype(self.data.dtype)

        # Transform annotations
        transformed_optic_disc = self._optic_disc.transform(annotation_matrix) if self._optic_disc is not None else None
        transformed_fovea = self._fovea.transform(annotation_matrix) if self._fovea is not None else None

        # Create new EyeEnface instance with updated metadata
        new_enface = EyeEnface(
            data=transformed_data,
            meta=updated_meta,
            optic_disc=transformed_optic_disc,
            fovea=transformed_fovea
        )

        # Transform area maps
        for area_map in self._area_maps:
            if area_map.data is not None:
                transformed_area_data = transform.warp(
                    area_map.data.astype(float),
                    tform.inverse,
                    output_shape=output_shape,
                    order=0,  # Use nearest neighbor for binary masks
                    mode=mode,
                    cval=cval,
                    preserve_range=True
                ).astype(bool)
            else:
                transformed_area_data = None

            new_area_map = EyeEnfacePixelAnnotation(
                new_enface,
                transformed_area_data,
                area_map.meta.copy()
            )
            new_enface._area_maps.append(new_area_map)

        return new_enface
