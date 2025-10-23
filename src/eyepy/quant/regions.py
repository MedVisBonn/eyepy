from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from eyepy.quant.metrics import compute_area
from eyepy.quant.spatial import _normalize_origin_mode
from eyepy.quant.spatial import AnatomicalOrigin
from eyepy.quant.spatial import DirectionalExtent
from eyepy.quant.spatial import OriginMode
from eyepy.quant.spatial import OriginModeType
from eyepy.quant.spatial import PolarReference

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from eyepy.core.eyeenface import EyeEnface


@dataclass
class RegionQuantification:
    """Base class for region quantification results.

    Attributes:
        area: Area of the region in square micrometers
        extent: Directional extent from anatomical origin
        origin: Anatomical origin used for directional measurements (optional, for plotting)
        mask: Binary mask of the region (optional, for plotting)
        scale_x: Micrometers per pixel in x-direction (optional, for plotting)
        scale_y: Micrometers per pixel in y-direction (optional, for plotting)
        unit: Unit for displaying distances, has to match the provided scale (Default: 'px')
    """

    area: float
    extent: DirectionalExtent
    origin: Optional[AnatomicalOrigin] = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    unit: str = 'px'

    def to_dict(self) -> dict:
        """Convert to dictionary for easy export.

        Returns:
            Dictionary with all quantification metrics
        """
        result = {
            'area': self.area,
        }
        result.update(self.extent.to_dict())
        return result

    def _format_distance(self, distance: float) -> str:
        """Format distance value according to the unit property.

        Args:
            distance: Distance value in micrometers

        Returns:
            Formatted string with value and unit
        """
        return f'{distance:.1f} {self.unit}'

    def plot(
        self,
        ax: Optional['Axes'] = None,
        show_origin: bool = True,
        show_extent_lines: bool = True,
        directions: Optional[list[str]] = None,
        origin_color: str = 'blue',
        origin_marker: str = 'x',
        origin_size: float = 100,
        line_color: str = 'yellow',
        line_width: float = 2,
        font_size: int = 10,
    ) -> 'Axes':
        """Plot the region with directional extent measurements.

        Visualizes the quantification by showing:
        - The anatomical origin point
        - Directional extent lines with distance annotations

        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            show_origin: Whether to mark the origin point
            show_extent_lines: Whether to show directional extent lines
            directions: List of directions to plot. If None, plots all 8 directions.
                       Valid values: 'temporal', 'nasal', 'superior', 'inferior',
                       'superior_temporal', 'inferior_temporal', 'superior_nasal',
                       'inferior_nasal'
            origin_color: Color for the origin marker
            origin_marker: Marker style for origin point
            origin_size: Size of origin marker
            line_color: Color for extent lines
            line_width: Width of extent lines
            font_size: Font size for distance annotations

        Returns:
            Matplotlib axes with the plot

        Raises:
            ValueError: If origin is not available when needed
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        # Check if origin is available for remaining visualizations
        if (show_origin or show_extent_lines) and self.origin is None:
            raise ValueError(
                'Origin must be provided to plot origin point or extent lines. '
                'Use from_mask() or from_mask_with_origin() to create quantification '
                'with origin information.'
            )

        # Show origin point
        if show_origin and self.origin is not None:
            ax.scatter(
                self.origin.x,
                self.origin.y,
                c=origin_color,
                marker=origin_marker,
                s=origin_size,
                zorder=10,
                label='Origin',
            )

        # Show directional extent lines
        if show_extent_lines and self.origin is not None:
            # Define all available directions and their angles
            all_directions = {
                'temporal': (0, 'T'),
                'nasal': (np.pi, 'N'),
                'superior': (-np.pi / 2, 'S'),
                'inferior': (np.pi / 2, 'I'),
                'superior_temporal': (-np.pi / 4, 'ST'),
                'inferior_temporal': (np.pi / 4, 'IT'),
                'superior_nasal': (-3 * np.pi / 4, 'SN'),
                'inferior_nasal': (3 * np.pi / 4, 'IN'),
            }

            # Filter to requested directions (or all if None)
            if directions is None:
                directions_to_plot = all_directions
            else:
                # Validate requested directions
                invalid = set(directions) - set(all_directions.keys())
                if invalid:
                    raise ValueError(
                        f'Invalid direction(s): {invalid}. '
                        f'Valid directions are: {list(all_directions.keys())}'
                    )
                directions_to_plot = {k: v for k, v in all_directions.items() if k in directions}

            for direction, (angle, label) in directions_to_plot.items():
                extent_metrics = getattr(self.extent, direction, None)
                if extent_metrics is None:
                    continue

                # Extract midpoint distance from ExtentMetrics
                distance = extent_metrics.midpoint
                if distance == 0:
                    continue

                # Convert angle and distance to Cartesian endpoint
                # Note: in image coordinates, y increases downward
                # angle=0 is temporal (positive x_cart)
                # We need to convert from anatomical coordinates back to image coordinates
                x_cart = distance * np.cos(angle)
                y_cart = distance * np.sin(angle)

                # Convert back to image coordinates from anatomical coordinates
                # For OD: x_cart positive = temporal = left = negative dx_image
                # For OS: x_cart positive = temporal = right = positive dx_image
                if self.origin.laterality == 'OD':
                    dx_image = -x_cart / self.scale_x
                else:
                    dx_image = x_cart / self.scale_x

                dy_image = y_cart / self.scale_y

                end_x = self.origin.x + dx_image
                end_y = self.origin.y + dy_image

                # Draw line from origin to endpoint
                ax.plot(
                    [self.origin.x, end_x],
                    [self.origin.y, end_y],
                    color=line_color,
                    linewidth=line_width,
                    zorder=5,
                )

                # Add text annotation at midpoint
                mid_x = (self.origin.x + end_x) / 2
                mid_y = (self.origin.y + end_y) / 2

                # Format distance based on unit
                dist_text = self._format_distance(distance)

                ax.text(
                    mid_x,
                    mid_y,
                    f'{label}: {dist_text}',
                    color='white',
                    fontsize=font_size,
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='black',
                        alpha=0.7,
                        edgecolor='none',
                    ),
                    zorder=15,
                )

        return ax

    @classmethod
    def from_mask_with_origin(
        cls,
        mask: npt.NDArray[np.bool_],
        origin: AnatomicalOrigin,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        unit: str = 'px',
    ):
        """Quantify region from a binary mask with custom origin.

        Computes directional extent (midpoint) and arc statistics in all 8 directions (temporal, nasal,
        superior, inferior, superior_temporal, inferior_temporal, superior_nasal,
        inferior_nasal).

        Args:
            mask: Binary mask of the region
            origin: AnatomicalOrigin defining the reference point
            scale_x: Size of a pixel in x-direction (default: 1.0)
            scale_y: Size of a pixel in y-direction (default: 1.0)
            unit: Unit for displaying distances (default: 'px')

        Returns:
            RegionQuantification subclass instance with computed metrics
        """
        # Compute area
        area = compute_area(mask, scale_x, scale_y)

        # Compute directional extent
        polar_ref = PolarReference(origin)
        extent = polar_ref.compute_directional_extent(
            mask,
            scale_x=scale_x,
            scale_y=scale_y,
        )

        return cls(
            area=area,
            extent=extent,
            origin=origin,
            scale_x=scale_x,
            scale_y=scale_y,
            unit=unit,
        )

    @classmethod
    def from_mask(
        cls,
        mask: npt.NDArray[np.bool_],
        optic_disc_center: Optional[tuple[float, float]] = None,
        fovea_center: Optional[tuple[float, float]] = None,
        laterality: Optional[str] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        origin_mode: OriginModeType = OriginMode.HYBRID,
        unit: str = 'px',
    ):
        """Quantify region from a binary mask.

        Computes directional extent (midpoint) and arc statistics in all 8 directions (temporal, nasal,
        superior, inferior, superior_temporal, inferior_temporal, superior_nasal,
        inferior_nasal) using both midpoint and statistics methods.

        Args:
            mask: Binary mask of the region
            optic_disc_center: (y, x) coordinates of optic disc center
            fovea_center: (y, x) coordinates of fovea center
            laterality: Eye laterality ('OD' or 'OS')
            scale_x: Micrometers per pixel in x-direction
            scale_y: Micrometers per pixel in y-direction
            origin_mode: Mode determining the reference origin.
                        Can be 'optic_disc', 'fovea', 'hybrid' or 'custom'
                        (default: 'hybrid')
            unit: Unit for displaying distances (default: 'px')

        Returns:
            RegionQuantification subclass instance with computed metrics
        """
        # Normalize origin_mode to ENUM
        origin_mode = _normalize_origin_mode(origin_mode)

        # Create anatomical origin based on mode
        if origin_mode == OriginMode.OPTIC_DISC:
            if optic_disc_center is None:
                raise ValueError(
                    'optic_disc_center must be provided when origin_mode is OPTIC_DISC.'
                )
            if laterality is None:
                raise ValueError(
                    'laterality must be provided when origin_mode is OPTIC_DISC.'
                )
            origin = AnatomicalOrigin.from_optic_disc(
                optic_disc_center=optic_disc_center,
                laterality=laterality,
            )
        elif origin_mode == OriginMode.FOVEA:
            if fovea_center is None:
                raise ValueError(
                    'fovea_center must be provided when origin_mode is FOVEA.'
                )
            if laterality is None:
                raise ValueError(
                    'laterality must be provided when origin_mode is FOVEA.'
                )
            origin = AnatomicalOrigin.from_fovea(
                fovea_center=fovea_center,
                laterality=laterality,
            )
        elif origin_mode == OriginMode.HYBRID:
            if optic_disc_center is None:
                raise ValueError(
                    'optic_disc_center must be provided when origin_mode is HYBRID.'
                )
            if fovea_center is None:
                raise ValueError(
                    'fovea_center must be provided when origin_mode is HYBRID.'
                )
            if laterality is None:
                raise ValueError(
                    'laterality must be provided when origin_mode is HYBRID.'
                )

            origin = AnatomicalOrigin.from_hybrid(
                optic_disc_center=optic_disc_center,
                fovea_center=fovea_center,
                laterality=laterality,
            )
        elif origin_mode == OriginMode.CUSTOM:
            raise ValueError(
                f'Use from_mask_with_origin() for OriginMode.CUSTOM'
            )

        return cls.from_mask_with_origin(
            mask=mask,
            origin=origin,
            scale_x=scale_x,
            scale_y=scale_y,
            unit=unit,
        )

    @classmethod
    def from_EyeEnface(
        cls,
        enface: 'EyeEnface',
        area_map_name: str,
        origin_mode: OriginModeType = OriginMode.HYBRID,
        custom_origin: Optional[AnatomicalOrigin] = None,
    ):
        """Create RegionQuantification from an EyeEnface object.

        A RegionQuantification quantifies an annotation mask of one connected component.
        Extents from a specified origin are computed in 8 directions.

        Args:
            enface: EyeEnface object containing the image and annotations
            area_map_name: Name of the area map containing the region to quantify
            origin_mode: Mode determining the reference origin.
                        Can be 'optic_disc', 'fovea', 'hybrid' or 'custom'
                        (default: 'hybrid')
            custom_origin: AnatomicalOrigin to use when origin_mode is 'custom'
                          (default: None)

        Returns:
            RegionQuantification subclass instance with computed metrics

        Raises:
            ValueError: If required annotations or area map are missing
        """
        # Normalize origin_mode to ENUM
        origin_mode = _normalize_origin_mode(origin_mode)

        # Validate based on origin_mode
        if origin_mode == OriginMode.CUSTOM:
            if custom_origin is None:
                raise ValueError(
                    'custom_origin must be provided when origin_mode is CUSTOM.'
                )
        elif origin_mode == OriginMode.OPTIC_DISC:
            if enface.optic_disc is None:
                raise ValueError(
                    'EyeEnface must have optic_disc annotation when origin_mode is OPTIC_DISC. '
                    'Set enface.optic_disc before calling from_EyeEnface().'
                )
        elif origin_mode == OriginMode.FOVEA:
            if enface.fovea is None:
                raise ValueError(
                    'EyeEnface must have fovea annotation when origin_mode is FOVEA. '
                    'Set enface.fovea before calling from_EyeEnface().'
                )
        elif origin_mode == OriginMode.HYBRID:
            if enface.optic_disc is None:
                raise ValueError(
                    'EyeEnface must have optic_disc annotation when origin_mode is HYBRID. '
                    'Set enface.optic_disc before calling from_EyeEnface().'
                )
            if enface.fovea is None:
                raise ValueError(
                    'EyeEnface must have fovea annotation when origin_mode is HYBRID. '
                    'Set enface.fovea before calling from_EyeEnface().'
                )

        # Get the area map
        if area_map_name not in enface.area_maps:
            raise ValueError(
                f'Area map "{area_map_name}" not found in EyeEnface. '
                f'Available area maps: {list(enface.area_maps.keys())}.'
            )

        area_annotation = enface.area_maps[area_map_name]
        mask = area_annotation.data

        # Get scales
        scale_x = enface.scale_x
        scale_y = enface.scale_y

        # Get unit from metadata if available, otherwise default to 'px'
        unit = enface.meta.get('scale_unit', 'px')

        # Handle different origin modes
        if origin_mode == OriginMode.CUSTOM:
            # custom_origin is guaranteed to be not None due to validation above
            assert custom_origin is not None
            return cls.from_mask_with_origin(
                mask=mask,
                origin=custom_origin,
                scale_x=scale_x,
                scale_y=scale_y,
                unit=unit,
            )
        else:
            # Check for laterality (needed for non-CUSTOM modes)
            laterality = enface.meta.get('laterality', None)
            if laterality is None:
                raise ValueError(
                    'EyeEnface must have laterality information in metadata '
                    f'when origin_mode is {origin_mode.value}. '
                    'Set enface.meta["laterality"] to "OD" or "OS".'
                )

            # Get anatomical landmarks (guaranteed to be not None due to validation above)
            optic_disc_center = enface.optic_disc.center if enface.optic_disc else (0.0, 0.0)
            fovea_center = enface.fovea.center if enface.fovea else (0.0, 0.0)

            # Create quantification using existing from_mask method
            return cls.from_mask(
                mask=mask,
                optic_disc_center=optic_disc_center,
                fovea_center=fovea_center,
                laterality=laterality,
                scale_x=scale_x,
                scale_y=scale_y,
                origin_mode=origin_mode,
                unit=unit,
            )
