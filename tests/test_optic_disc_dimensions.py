"""Tests for optic disc width and height measurement methods."""
import numpy as np
import pytest

from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation


class TestOpticDiscDimensions:
    """Test dimension measurement methods for optic disc using horizontal/vertical
    extent."""

    def test_width_height_ellipse_from_circle(self):
        """Test that width/height work correctly for a circle."""
        # Create a circular optic disc
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=30,
            height=30,  # Same as width -> circle
            rotation=0,
            shape=(200, 200)
        )

        # For a circle, width and height should be approximately equal
        assert abs(optic_disc.width - 30) < 2, 'Width should be approximately 30'
        assert abs(optic_disc.height - 30) < 2, 'Height should be approximately 30'
        assert abs(optic_disc.width - optic_disc.height) < 2, 'Width and height should be nearly equal for a circle'

    def test_width_height_ellipse_from_ellipse(self):
        """Test that width/height work correctly for an ellipse."""
        # Create an elliptical optic disc
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=25,
            height=35,  # Taller than wide
            rotation=0,
            shape=(200, 200)
        )

        # For an axis-aligned ellipse, width should match horizontal extent
        # and height should match vertical extent
        assert optic_disc.width < optic_disc.height, 'Width should be less than height for vertical ellipse'
        assert abs(optic_disc.width - 25) < 3, 'Width should be approximately 25'
        assert abs(optic_disc.height - 35) < 3, 'Height should be approximately 35'

    def test_width_height_from_circle(self):
        """Test horizontal/vertical dimensions for a circular optic disc."""
        # Create a circular optic disc
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=30,
            height=30,
            rotation=0,
            shape=(200, 200)
        )

        # For a circle, horizontal and vertical extents should be approximately equal
        width = optic_disc.width
        height = optic_disc.height

        assert abs(width - 30) < 2, f"Horizontal extent should be approximately 30, got {width}"
        assert abs(height - 30) < 2, f"Vertical extent should be approximately 30, got {height}"
        assert abs(width - height) < 2, 'Horizontal and vertical extents should be nearly equal for a circle'

    def test_width_height_from_ellipse_no_rotation(self):
        """Test horizontal/vertical dimensions for an axis-aligned ellipse."""
        # Create an ellipse with no rotation
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=25,   # Horizontal extent
            height=35,  # Vertical extent
            rotation=0,
            shape=(200, 200)
        )

        width = optic_disc.width
        height = optic_disc.height

        # For an unrotated ellipse, width should match width, height should match height
        assert abs(width - 25) < 3, f"Horizontal extent should be approximately 25, got {width}"
        assert abs(height - 35) < 3, f"Vertical extent should be approximately 35, got {height}"

    def test_width_height_from_rotated_ellipse(self):
        """Test that horizontal/vertical dimensions change with rotation."""
        # Create an ellipse rotated 45 degrees
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=20,
            height=40,
            rotation=np.pi/4,  # 45 degrees
            shape=(200, 200)
        )

        width = optic_disc.width
        height = optic_disc.height

        # After 45-degree rotation, horizontal and vertical extents should be more similar
        # (the elongated ellipse is now diagonal)
        assert width > 20, 'Horizontal extent should be larger than original width due to rotation'
        assert height > 20, 'Vertical extent should be larger than original width due to rotation'

        # They should be approximately equal for 45-degree rotation of a 1:2 aspect ratio
        assert abs(width - height) < 5, 'Horizontal and vertical extents should be similar after 45° rotation'

    def test_width_height_90_degree_rotation(self):
        """Test that 90-degree rotation swaps horizontal and vertical extents."""
        # Create an ellipse with width < height
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=25,
            height=35,
            rotation=0,
            shape=(200, 200)
        )

        width_0 = optic_disc.width
        height_0 = optic_disc.height

        # Rotate 90 degrees
        optic_disc_90 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=25,
            height=35,
            rotation=np.pi/2,  # 90 degrees
            shape=(200, 200)
        )

        width_90 = optic_disc_90.width
        height_90 = optic_disc_90.height

        # After 90-degree rotation, horizontal and vertical should swap
        assert abs(width_0 - height_90) < 3, 'Original width should approximately equal rotated height'
        assert abs(height_0 - width_90) < 3, 'Original height should approximately equal rotated width'

    def test_dimensions_similar_for_rotated(self):
        """Test that dimensions are reasonable for rotated ellipses."""
        # Create a rotated ellipse
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100),
            width=20,
            height=40,
            rotation=np.pi/6,  # 30 degrees
            shape=(200, 200)
        )

        # Dimensions should be reasonable
        width = optic_disc.width
        height = optic_disc.height

        # Width and height should still reflect that this is a taller structure
        assert width < height, 'Width should be less than height'
        # Both should be between the original dimensions
        assert 20 <= width <= 40, f"Width should be between 20 and 40, got {width}"
        assert 20 <= height <= 40, f"Height should be between 20 and 40, got {height}"

    def test_center_consistency_across_measurements(self):
        """Test that all measurements use the same center point."""
        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 150),
            width=25,
            height=35,
            rotation=0.3,
            shape=(200, 200)
        )

        # The center should be the same for all measurements
        center = optic_disc.center
        assert abs(center[0] - 100) < 1, 'Center y should be approximately 100'
        assert abs(center[1] - 150) < 1, 'Center x should be approximately 150'

    def test_dimensions_for_irregular_polygon(self):
        """Test dimensions for a manually created irregular polygon."""
        # Create an irregular polygon (approximate rectangle)
        polygon = np.array([
            [90, 80],   # Top-left
            [90, 120],  # Top-right
            [110, 120], # Bottom-right
            [110, 80],  # Bottom-left
        ])

        optic_disc = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(200, 200))

        # H/V dimensions should approximately match the rectangle
        width = optic_disc.width
        height = optic_disc.height

        # Rectangle is 40 wide (x: 80-120) and 20 tall (y: 90-110)
        assert abs(width - 40) < 2, f"Horizontal extent should be approximately 40, got {width}"
        assert abs(height - 20) < 2, f"Vertical extent should be approximately 20, got {height}"

    def test_dimensions_change_with_rotation(self):
        """Test that dimensions change as expected with rotation."""
        width, height = 20, 40

        # At 0 rotation: width ≈ width, height ≈ height
        optic_disc_0 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=0, shape=(200, 200)
        )

        # At 90 rotation: dimensions should swap
        optic_disc_90 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=np.pi/2, shape=(200, 200)
        )

        # At 45 rotation: dimensions should be more similar
        optic_disc_45 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=np.pi/4, shape=(200, 200)
        )

        # Check that 0-degree has width < height
        assert optic_disc_0.width < optic_disc_0.height

        # Check that 90-degree has width > height (swapped)
        assert optic_disc_90.width > optic_disc_90.height

        # Check that 45-degree has more similar dimensions
        ratio_0 = optic_disc_0.height / optic_disc_0.width
        ratio_45 = optic_disc_45.height / optic_disc_45.width
        assert ratio_45 < ratio_0, '45-degree rotation should make dimensions more similar'

    def test_hv_dimensions_change_with_rotation(self):
        """Test that h/v dimensions change as expected with rotation."""
        width, height = 20, 40

        # At 0 rotation: width ≈ width, height ≈ height
        optic_disc_0 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=0, shape=(200, 200)
        )

        # At 90° rotation: they should swap
        optic_disc_90 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=np.pi/2, shape=(200, 200)
        )

        # At 45° rotation: they should be more similar
        optic_disc_45 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(100, 100), width=width, height=height,
            rotation=np.pi/4, shape=(200, 200)
        )

        # Check that rotation affects h/v dimensions
        diff_0 = abs(optic_disc_0.width - optic_disc_0.height)
        diff_45 = abs(optic_disc_45.width - optic_disc_45.height)
        diff_90 = abs(optic_disc_90.width - optic_disc_90.height)

        # At 45 degrees, the difference should be smallest
        assert diff_45 < diff_0, 'Difference should decrease at 45° rotation'
        assert diff_45 < diff_90, 'Difference should be smallest at 45° rotation'
