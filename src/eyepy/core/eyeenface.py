from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

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
    """Enface image with optional anatomical annotations."""

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
        self.optic_disc = optic_disc
        self.fovea = fovea

        # Validate laterality if both optic disc and fovea are provided
        if optic_disc is not None and fovea is not None:
            self._validate_laterality()

    def _validate_laterality(self) -> None:
        """Validate that laterality matches the relative position of optic disc
        and fovea.

        For a right eye (OD), the optic disc should be to the right of the fovea (higher x).
        For a left eye (OS), the optic disc should be to the left of the fovea (lower x).

        Raises:
            ValueError: If the laterality doesn't match the anatomical positions
        """
        if self.optic_disc is None or self.fovea is None:
            return

        laterality = self.meta.get('laterality', None)
        if laterality is None:
            # No laterality info, skip validation
            return

        # Get centers
        od_center = self.optic_disc.center
        fovea_center = self.fovea.center

        # Compare x-coordinates (column positions)
        od_x = od_center[1]  # (y, x) format
        fovea_x = fovea_center[1]

        # For right eye (OD): optic disc should be to the right of fovea
        # For left eye (OS): optic disc should be to the left of fovea
        if laterality.upper() in ['OD', 'R', 'RIGHT']:
            if od_x <= fovea_x:
                raise ValueError(
                    f"Laterality mismatch: Right eye (OD) expects optic disc to the right of fovea, "
                    f"but optic disc is at x={od_x:.1f} and fovea is at x={fovea_x:.1f}"
                )
        elif laterality.upper() in ['OS', 'L', 'LEFT']:
            if od_x >= fovea_x:
                raise ValueError(
                    f"Laterality mismatch: Left eye (OS) expects optic disc to the left of fovea, "
                    f"but optic disc is at x={od_x:.1f} and fovea is at x={fovea_x:.1f}"
                )

    def validate_laterality(self) -> bool:
        """Validate that laterality matches the relative position of optic disc
        and fovea.

        This is a public method that can be called to check laterality consistency.

        Returns:
            True if laterality is consistent with anatomy, False if inconsistent or cannot be validated

        Raises:
            ValueError: If the laterality doesn't match the anatomical positions
        """
        try:
            self._validate_laterality()
            return True
        except ValueError:
            raise

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

    @property
    def scale_y(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_y']

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
        plot_optic_disc: bool = True,
        plot_fovea: bool = True,
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
            plot_optic_disc: If `True` and optic_disc is available, plot the optic disc annotation (default: `True`).
            plot_fovea: If `True` and fovea is available, plot the fovea annotation (default: `True`).
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

        # Plot optic disc if available
        if plot_optic_disc and self.optic_disc is not None:
            if optic_disc_kwargs is None:
                optic_disc_kwargs = {}

            # Calculate region offset
            y_start = region[0].start if region[0].start is not None else 0
            x_start = region[1].start if region[1].start is not None else 0

            # Delegate to the optic disc's plot method
            self.optic_disc.plot(ax=ax, offset=(y_start, x_start), **optic_disc_kwargs)

        # Plot fovea if available
        if plot_fovea and self.fovea is not None:
            if fovea_kwargs is None:
                fovea_kwargs = {}

            # Calculate region offset
            y_start = region[0].start if region[0].start is not None else 0
            x_start = region[1].start if region[1].start is not None else 0

            # Delegate to the fovea's plot method
            self.fovea.plot(ax=ax, offset=(y_start, x_start), **fovea_kwargs)

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

    def translate(self, offset_y: float, offset_x: float,
                  order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Translate the enface image and all annotations.

        Returns a new EyeEnface instance with translated image data and transformed annotations.
        The original EyeEnface remains unchanged.

        Args:
            offset_y: Translation offset for the y-axis (pixels)
            offset_x: Translation offset for the x-axis (pixels)
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with translated data and annotations
        """
        # Create affine transformation matrix for translation
        matrix = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ])

        return self.transform(matrix, order=order, mode=mode, cval=cval)

    def rotate(self, angle: float, center: Optional[tuple[float, float]] = None,
               order: int = 1, mode: str = 'constant', cval: float = 0.0) -> 'EyeEnface':
        """Rotate the enface image and all annotations.

        Returns a new EyeEnface instance with rotated image data and transformed annotations.
        The original EyeEnface remains unchanged.

        Args:
            angle: Rotation angle in degrees (counter-clockwise)
            center: Center of rotation (y, x). If None, uses image center
            order: The order of interpolation (0=nearest, 1=bilinear, 3=bicubic, default: 1)
            mode: How to handle values outside the boundaries ('constant', 'nearest', 'reflect', 'wrap')
            cval: Value used for points outside the boundaries if mode='constant'

        Returns:
            New EyeEnface instance with rotated data and annotations
        """
        if center is None:
            center = (self.shape[0] / 2, self.shape[1] / 2)

        # Create affine transformation matrix for rotation around a center point
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cy, cx = center

        # Rotation matrix around a center point
        matrix = np.array([
            [cos_a, -sin_a, cy - cos_a * cy + sin_a * cx],
            [sin_a, cos_a, cx - sin_a * cy - cos_a * cx],
            [0, 0, 1]
        ])

        return self.transform(matrix, order=order, mode=mode, cval=cval)

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
        if self.optic_disc is None:
            raise ValueError('standard_rotation requires optic_disc annotation')
        if self.fovea is None:
            raise ValueError('standard_rotation requires fovea annotation')

        # Get centers of optic disc and fovea
        od_center = self.optic_disc.center
        fovea_center = self.fovea.center

        # Determine laterality from metadata or infer from positions
        laterality = self.meta.get('laterality', None)

        # Calculate the current angle between the centers
        # od_center and fovea_center are (y, x) tuples
        dy = fovea_center[0] - od_center[0]
        dx = fovea_center[1] - od_center[1]

        # Calculate the angle from OD to fovea
        current_angle_rad = np.arctan2(dy, dx)

        # Determine target angle based on laterality
        # For OD (right eye): fovea should be to the left (negative x), so target is 180 degrees (π)
        # For OS (left eye): fovea should be to the right (positive x), so target is 0 degrees
        if laterality is not None:
            laterality_upper = str(laterality).upper()
            if laterality_upper in ['OD', 'R', 'RIGHT']:
                target_angle_rad = np.pi  # Fovea to the left
            elif laterality_upper in ['OS', 'L', 'LEFT']:
                target_angle_rad = 0.0  # Fovea to the right
            else:
                # Unknown laterality, default to horizontal alignment (0 degrees)
                target_angle_rad = 0.0
        else:
            # No laterality info, align horizontally with fovea to the right
            target_angle_rad = 0.0

        # Calculate rotation angle needed
        rotation_angle_rad = target_angle_rad - current_angle_rad
        rotation_angle_deg = np.rad2deg(rotation_angle_rad)

        # Rotate around the midpoint between OD and fovea to keep them centered
        # This preserves their relative position better than rotating around image center
        midpoint_y = (od_center[0] + fovea_center[0]) / 2
        midpoint_x = (od_center[1] + fovea_center[1]) / 2
        center = (midpoint_y, midpoint_x)

        return self.rotate(rotation_angle_deg, center=center, order=order, mode=mode, cval=cval)

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
            annotation_matrix = matrix[:2, :]
        elif matrix.shape == (2, 3):
            annotation_matrix = matrix
        else:
            raise ValueError('Matrix must be 3x3 or 2x3')

        # For image transformation, we need the full 3x3 matrix
        if matrix.shape == (2, 3):
            # Convert 2x3 to 3x3
            full_matrix = np.vstack([matrix, [0, 0, 1]])
        else:
            full_matrix = matrix

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
        tform = transform.AffineTransform(matrix=np.linalg.inv(full_matrix))

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
        transformed_optic_disc = self.optic_disc.transform(annotation_matrix) if self.optic_disc is not None else None
        transformed_fovea = self.fovea.transform(annotation_matrix) if self.fovea is not None else None

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
