"""Spatial reference systems for anatomical quantification.

This module provides classes for defining spatial reference frames based
on anatomical landmarks (optic disc, fovea) and computing distances and
directions relative to these references.
"""

from dataclasses import dataclass
from enum import Enum
from typing import cast, Literal, Optional, Union

import numpy as np
import numpy.typing as npt


class OriginMode(Enum):
    """Mode for determining the anatomical origin.

    OPTIC_DISC: Use optic disc center as origin
    FOVEA: Use fovea center as origin
    HYBRID: Use optic disc x-coordinate and fovea y-coordinate
    CUSTOM: Use a custom user-specified origin point
    """

    OPTIC_DISC = 'optic_disc'
    FOVEA = 'fovea'
    HYBRID = 'hybrid'
    CUSTOM = 'custom'


# Type alias for flexible API - accept both ENUM and strings
OriginModeType = Union[OriginMode, Literal['optic_disc', 'fovea', 'hybrid', 'custom']]


def _normalize_origin_mode(mode: OriginModeType) -> OriginMode:
    """Convert string to OriginMode ENUM if needed.

    Args:
        mode: Either an OriginMode enum or a string literal

    Returns:
        OriginMode enum

    Raises:
        ValueError: If string mode is not a valid option
    """
    if isinstance(mode, str):
        try:
            return OriginMode(mode)
        except ValueError:
            valid_values = [m.value for m in OriginMode]
            raise ValueError(
                f"Invalid origin_mode: '{mode}'. "
                f'Must be one of {valid_values} or use OriginMode enum'
            )
    return mode


@dataclass
class AnatomicalOrigin:
    """Reference origin based on anatomical landmarks.

    Defines a coordinate system origin that can be based on the optic disc,
    fovea, a hybrid approach, or a custom position.

    **Coordinate System Convention:**
    This class uses (row, col) image coordinates for input and output:
    - row: vertical axis, increases downward (corresponds to y)
    - col: horizontal axis, increases rightward (corresponds to x)

    The internal storage uses (y, x) where y=row and x=col.
    All methods that accept or return coordinates use (row, col) format unless
    explicitly documented otherwise.

    Attributes:
        y: Vertical (y) coordinate of the origin
        x: Horizontal (x) coordinate of the origin
        laterality: Eye laterality ('OD' or 'OS')
        mode: Origin mode used to determine the position
    """

    y: float
    x: float
    laterality: str
    mode: OriginMode

    @classmethod
    def from_optic_disc(
        cls,
        optic_disc_center: tuple[float, float],
        laterality: str,
    ) -> 'AnatomicalOrigin':
        """Create origin at optic disc center.

        Args:
            optic_disc_center: (y, x) coordinates of optic disc center
            laterality: Eye laterality ('OD' or 'OS')

        Returns:
            AnatomicalOrigin at optic disc center
        """
        if laterality not in ['OD', 'OS']:
            raise ValueError(f'Laterality must be OD or OS, got {laterality}')

        return cls(
            y=optic_disc_center[0],
            x=optic_disc_center[1],
            laterality=laterality,
            mode=OriginMode.OPTIC_DISC,
        )

    @classmethod
    def from_fovea(
        cls,
        fovea_center: tuple[float, float],
        laterality: str,
    ) -> 'AnatomicalOrigin':
        """Create origin at fovea center.

        Args:
            fovea_center: (y, x) coordinates of fovea center
            laterality: Eye laterality ('OD' or 'OS')

        Returns:
            AnatomicalOrigin at fovea center
        """
        if laterality not in ['OD', 'OS']:
            raise ValueError(f'Laterality must be OD or OS, got {laterality}')

        return cls(
            y=fovea_center[0],
            x=fovea_center[1],
            laterality=laterality,
            mode=OriginMode.FOVEA,
        )

    @classmethod
    def from_hybrid(
        cls,
        optic_disc_center: tuple[float, float],
        fovea_center: tuple[float, float],
        laterality: str,
    ) -> 'AnatomicalOrigin':
        """Create hybrid origin from optic disc and fovea positions.

        Uses the horizontal (x) position from the optic disc center and
        the vertical (y) position from the fovea center.

        Args:
            optic_disc_center: (y, x) coordinates of optic disc center
            fovea_center: (y, x) coordinates of fovea center
            laterality: Eye laterality ('OD' or 'OS')

        Returns:
            AnatomicalOrigin with y from fovea, x from optic disc
        """
        if laterality not in ['OD', 'OS']:
            raise ValueError(f'Laterality must be OD or OS, got {laterality}')

        # Origin: horizontal position from OD, vertical position from fovea
        origin_y = fovea_center[0]
        origin_x = optic_disc_center[1]

        return cls(
            y=origin_y,
            x=origin_x,
            laterality=laterality,
            mode=OriginMode.HYBRID,
        )

    @classmethod
    def from_custom(
        cls,
        origin: tuple[float, float],
        laterality: str,
    ) -> 'AnatomicalOrigin':
        """Create origin at custom position.

        Args:
            origin: (y, x) coordinates of custom origin
            laterality: Eye laterality ('OD' or 'OS')

        Returns:
            AnatomicalOrigin at custom position
        """
        if laterality not in ['OD', 'OS']:
            raise ValueError(f'Laterality must be OD or OS, got {laterality}')

        return cls(
            y=origin[0],
            x=origin[1],
            laterality=laterality,
            mode=OriginMode.CUSTOM,
        )

    def to_cartesian(self, y: float, x: float) -> tuple[float, float]:
        """Convert image coordinates (row, col) to Cartesian coordinates relative
        to origin.

        Image coordinates use (row, col) convention where:
        - row increases downward
        - col increases to the right

        Cartesian output coordinates are anatomically oriented:
        - x: horizontal (positive = temporal, negative = nasal)
        - y: vertical (positive = inferior, negative = superior)

        Args:
            y: Image row coordinate (increases downward)
            x: Image column coordinate (increases to the right)

        Returns:
            (x_cart, y_cart) Cartesian coordinates relative to origin where:
            - x_cart: horizontal distance (positive = temporal, negative = nasal)
            - y_cart: vertical distance (positive = inferior/downward, negative = superior/upward)
        """
        # Compute displacement in image coordinates
        dx_image = x - self.x  # Column displacement
        dy_image = y - self.y  # Row displacement (positive = downward)

        # In Cartesian coordinates:
        # - y_cart = dy_image (positive downward = inferior, negative upward = superior)
        # - x_cart needs laterality adjustment for temporal/nasal

        y_cart = dy_image  # Positive = inferior/downward, negative = superior/upward

        # Adjust for laterality: temporal direction
        # OD (right eye): nasal is to the right (positive dx), temporal is to the left (negative dx)
        #                 So temporal (positive x_cart) = negative dx_image
        # OS (left eye): nasal is to the left (negative dx), temporal is to the right (positive dx)
        #                So temporal (positive x_cart) = positive dx_image
        if self.laterality == 'OD':
            x_cart = -dx_image
        else:  # OS
            x_cart = dx_image

        return (x_cart, y_cart)

    def to_polar(self, y: float, x: float) -> tuple[float, float]:
        """Convert image coordinates (row, col) to polar coordinates.

        Args:
            y: Image row coordinate (increases downward)
            x: Image column coordinate (increases to the right)

        Returns:
            (distance, angle) where:
            - distance: Euclidean distance from origin
            - angle: Angle in radians (0 = temporal, π/2 = inferior,
                     π = nasal, 3π/2 = superior)
        """
        x_cart, y_cart = self.to_cartesian(y, x)
        distance = np.sqrt(x_cart**2 + y_cart**2)
        angle = np.arctan2(y_cart, x_cart)  # Range: [-π, π]

        # Convert to [0, 2π] with 0 = temporal
        if angle < 0:
            angle += 2 * np.pi

        return (distance, angle)


@dataclass
class ExtentMetrics:
    """Metrics for extent in a single direction.

    Contains statistical measures of distances from the origin to boundary
    points within a single angular sector.

    Attributes:
        midpoint: Distance to boundary at exact midpoint angle of this direction
        mean: Mean distance to boundary points in this direction
        max: Maximum distance to boundary in this direction
        median: Median distance to boundary points in this direction
        std: Standard deviation of distances in this direction
        touches_border: Whether the region extends to the image border in this
                       specific direction, indicating measurements may be truncated
    """

    midpoint: float
    mean: float
    max: float
    median: float
    std: float
    touches_border: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary mapping metric names to values
        """
        return {
            'midpoint': self.midpoint,
            'mean': self.mean,
            'max': self.max,
            'median': self.median,
            'std': self.std,
            'touches_border': self.touches_border,
        }


@dataclass
class DirectionalExtent:
    """Extent of a region in specific directions from origin.

    Contains ExtentMetrics for each of the 8 anatomical directions (4 cardinal
    + 4 ordinal) from an anatomical origin. Each direction includes mean, max,
    median, and standard deviation of boundary distances, along with a flag
    indicating if that specific direction touches the image border.

    Attributes:
        temporal: Extent metrics in temporal direction
        nasal: Extent metrics in nasal direction
        superior: Extent metrics in superior direction
        inferior: Extent metrics in inferior direction
        superior_temporal: Extent metrics in superior-temporal direction
        inferior_temporal: Extent metrics in inferior-temporal direction
        superior_nasal: Extent metrics in superior-nasal direction
        inferior_nasal: Extent metrics in inferior-nasal direction
    """

    temporal: ExtentMetrics
    nasal: ExtentMetrics
    superior: ExtentMetrics
    inferior: ExtentMetrics
    superior_temporal: ExtentMetrics
    inferior_temporal: ExtentMetrics
    superior_nasal: ExtentMetrics
    inferior_nasal: ExtentMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary with nested structure: direction -> metric -> value
        """
        return {
            'temporal': self.temporal.to_dict(),
            'nasal': self.nasal.to_dict(),
            'superior': self.superior.to_dict(),
            'inferior': self.inferior.to_dict(),
            'superior_temporal': self.superior_temporal.to_dict(),
            'inferior_temporal': self.inferior_temporal.to_dict(),
            'superior_nasal': self.superior_nasal.to_dict(),
            'inferior_nasal': self.inferior_nasal.to_dict(),
        }


def _extract_boundary_with_border(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Extract boundary pixels including those at image borders.

    This function properly handles masks that extend to the image border,
    ensuring the outer boundary is captured even when the region touches edges.

    Args:
        mask: Binary mask of the region

    Returns:
        y_coords: Y coordinates of boundary pixels
        x_coords: X coordinates of boundary pixels
        touches_border: Whether the region touches the image border
    """
    from scipy import ndimage
    from skimage import morphology

    # Fill holes to get outer boundary only (ignore inner boundaries)
    filled_mask = ndimage.binary_fill_holes(mask)
    assert filled_mask is not None  # Type narrowing

    # Check if region touches the border
    h, w = filled_mask.shape
    touches_border = (
        np.any(filled_mask[0, :]) or  # Top edge
        np.any(filled_mask[-1, :]) or  # Bottom edge
        np.any(filled_mask[:, 0]) or  # Left edge
        np.any(filled_mask[:, -1])     # Right edge
    )

    if touches_border:
        # Pad the mask to avoid losing border pixels during erosion
        padded_mask = np.pad(filled_mask, pad_width=1, mode='constant', constant_values=False)

        # Extract boundary from padded mask
        eroded = morphology.binary_erosion(padded_mask)
        boundary_padded = padded_mask & ~eroded

        # Remove padding (shift coordinates back)
        boundary = boundary_padded[1:-1, 1:-1]
    else:
        # Normal case: erosion works correctly
        eroded = morphology.binary_erosion(filled_mask)
        boundary = filled_mask & ~eroded

    # Get coordinates of boundary pixels
    y_coords, x_coords = np.where(boundary)

    return y_coords, x_coords, bool(touches_border)


class PolarReference:
    """Polar coordinate reference system for spatial analysis.

    Divides the image into 8 angular sectors (4 cardinal + 4 ordinal directions)
    relative to an anatomical origin for computing directional statistics.

    Attributes:
        origin: Anatomical origin point
    """

    def __init__(
        self,
        origin: AnatomicalOrigin,
    ):
        """Initialize polar reference system.

        Args:
            origin: Anatomical origin defining the coordinate system
        """
        self.origin = origin

    def compute_directional_extent(
        self,
        mask: npt.NDArray[np.bool_],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> DirectionalExtent:
        """Compute extent in all 8 directions with complete metrics.

        For each of the 8 anatomical directions (4 cardinal + 4 ordinal),
        computes:
        - Midpoint distance: distance to boundary at exact direction angle
        - Mean, max, median, std: statistics of all boundary points in that sector
        - Border flag: whether that specific direction touches the image border

        Args:
            mask: Binary mask of the region
            scale_x: Micrometers per pixel in x-direction
            scale_y: Micrometers per pixel in y-direction

        Returns:
            DirectionalExtent with ExtentMetrics for all 8 directions
        """
        # Define the 8 directions with their midpoint angles and sector ranges
        # Each sector spans π/4 radians (45 degrees) centered on its midpoint
        directions_config = {
            'temporal': {
                'midpoint_angle': 0.0,
                'sector_range': (-np.pi / 8, np.pi / 8),
            },
            'inferior_temporal': {
                'midpoint_angle': np.pi / 4,
                'sector_range': (np.pi / 8, 3 * np.pi / 8),
            },
            'inferior': {
                'midpoint_angle': np.pi / 2,
                'sector_range': (3 * np.pi / 8, 5 * np.pi / 8),
            },
            'inferior_nasal': {
                'midpoint_angle': 3 * np.pi / 4,
                'sector_range': (5 * np.pi / 8, 7 * np.pi / 8),
            },
            'nasal': {
                'midpoint_angle': np.pi,
                'sector_range': ((7 * np.pi / 8, np.pi), (-np.pi, -7 * np.pi / 8)),
            },
            'superior_nasal': {
                'midpoint_angle': -3 * np.pi / 4,
                'sector_range': (-7 * np.pi / 8, -5 * np.pi / 8),
            },
            'superior': {
                'midpoint_angle': -np.pi / 2,
                'sector_range': (-5 * np.pi / 8, -3 * np.pi / 8),
            },
            'superior_temporal': {
                'midpoint_angle': -np.pi / 4,
                'sector_range': (-3 * np.pi / 8, -np.pi / 8),
            },
        }

        if not np.any(mask):
            # Empty mask - return zeros for all directions
            zero_metrics = ExtentMetrics(
                midpoint=0.0, mean=0.0, max=0.0, median=0.0, std=0.0, touches_border=False
            )
            return DirectionalExtent(
                temporal=zero_metrics,
                nasal=zero_metrics,
                superior=zero_metrics,
                inferior=zero_metrics,
                superior_temporal=zero_metrics,
                inferior_temporal=zero_metrics,
                superior_nasal=zero_metrics,
                inferior_nasal=zero_metrics,
            )

        # Extract boundary pixels, properly handling border cases
        y_coords, x_coords, _ = _extract_boundary_with_border(mask)

        if len(y_coords) == 0:
            # Single pixel mask - use the pixel itself
            y_coords, x_coords = np.where(mask)

        # Compute metrics for all 8 directions
        mask_shape = (mask.shape[0], mask.shape[1])
        metrics = {}
        for direction_name, config in directions_config.items():
            metrics[direction_name] = self._compute_single_direction_metrics(
                y_coords=y_coords,
                x_coords=x_coords,
                scale_x=scale_x,
                scale_y=scale_y,
                midpoint_angle=config['midpoint_angle'],
                sector_range=config['sector_range'],
                mask_shape=mask_shape,
            )

        return DirectionalExtent(
            temporal=metrics['temporal'],
            nasal=metrics['nasal'],
            superior=metrics['superior'],
            inferior=metrics['inferior'],
            superior_temporal=metrics['superior_temporal'],
            inferior_temporal=metrics['inferior_temporal'],
            superior_nasal=metrics['superior_nasal'],
            inferior_nasal=metrics['inferior_nasal'],
        )

    def _compute_single_direction_metrics(
        self,
        y_coords: npt.NDArray,
        x_coords: npt.NDArray,
        scale_x: float,
        scale_y: float,
        midpoint_angle: float,
        sector_range: Union[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]],
        mask_shape: tuple[int, int],
    ) -> ExtentMetrics:
        """Compute all metrics for a single direction.

        Args:
            y_coords: Y coordinates of boundary pixels
            x_coords: X coordinates of boundary pixels
            scale_x: Micrometers per pixel in x-direction
            scale_y: Micrometers per pixel in y-direction
            midpoint_angle: Exact angle (in radians) for midpoint measurement
            sector_range: Angular range for this sector (angle_min, angle_max)
                         or ((angle_min1, angle_max1), (angle_min2, angle_max2)) for wraparound
            mask_shape: Shape of the mask (height, width) for border detection

        Returns:
            ExtentMetrics with all computed metrics for this direction
        """
        # Collect distances and boundary points in this sector
        distances = []
        sector_boundary_points = []

        for y, x in zip(y_coords, x_coords):
            x_cart, y_cart = self.origin.to_cartesian(y, x)

            # Skip origin point
            if x_cart == 0 and y_cart == 0:
                continue

            # Compute angle
            angle = np.arctan2(y_cart, x_cart)  # Range: [-π, π]

            # Check if point is in this sector
            in_sector = False
            if isinstance(sector_range[0], tuple):
                # Wraparound case (e.g., nasal direction)
                # sector_range is ((angle_min1, angle_max1), (angle_min2, angle_max2))
                wraparound_ranges = cast(tuple[tuple[float, float], tuple[float, float]], sector_range)
                in_sector = (angle >= wraparound_ranges[0][0]) or (angle <= wraparound_ranges[1][1])
            else:
                # Normal case: sector_range is (angle_min, angle_max)
                simple_range = cast(tuple[float, float], sector_range)
                in_sector = simple_range[0] <= angle < simple_range[1]

            if in_sector:
                # Compute scaled distance
                x_cart_scaled = x_cart * scale_x
                y_cart_scaled = y_cart * scale_y
                distance = np.sqrt(x_cart_scaled**2 + y_cart_scaled**2)
                distances.append(distance)
                sector_boundary_points.append((y, x))

        # Compute statistics
        if len(distances) > 0:
            mean_dist = float(np.mean(distances))
            max_dist = float(np.max(distances))
            median_dist = float(np.median(distances))
            std_dist = float(np.std(distances))
        else:
            mean_dist = max_dist = median_dist = std_dist = 0.0

        # Compute midpoint distance (distance at exact midpoint angle)
        # Find boundary point in this sector closest to the midpoint angle
        midpoint_dist = 0.0
        # Use sector_boundary_points if available (normal case)
        points_to_search = sector_boundary_points
        if len(points_to_search) > 0:
            min_angle_diff = float('inf')
            for y, x in points_to_search:
                x_cart, y_cart = self.origin.to_cartesian(y, x)
                if x_cart == 0 and y_cart == 0:
                    continue

                angle = np.arctan2(y_cart, x_cart)
                angle_diff = abs(angle - midpoint_angle)

                # Handle wraparound at ±π
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    x_cart_scaled = x_cart * scale_x
                    y_cart_scaled = y_cart * scale_y
                    midpoint_dist = float(np.sqrt(x_cart_scaled**2 + y_cart_scaled**2))

        # Check if this direction touches the border
        touches_border = False
        h, w = mask_shape
        for y, x in sector_boundary_points:
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                touches_border = True
                break

        return ExtentMetrics(
            midpoint=midpoint_dist,
            mean=mean_dist,
            max=max_dist,
            median=median_dist,
            std=std_dist,
            touches_border=touches_border,
        )
