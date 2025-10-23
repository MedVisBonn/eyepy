"""Unit tests for the EyeEnfaceOpticDiscAnnotation class."""
import numpy as np
import pytest

from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation


class TestOpticDisc:
    """Test suite for EyeEnfaceOpticDiscAnnotation class."""

    def test_init_from_polygon(self):
        """Test initialization from polygon."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        assert od.polygon.shape == (4, 2)
        assert od.shape == (50, 50)
        assert isinstance(od.polygon, np.ndarray)

    def test_init_validates_polygon_shape(self):
        """Test that polygon shape is validated."""
        # Wrong dimensions
        with pytest.raises(ValueError, match='Nx2 array'):
            EyeEnfaceOpticDiscAnnotation(polygon=np.array([1, 2, 3]))

        # Wrong number of columns
        with pytest.raises(ValueError, match='Nx2 array'):
            EyeEnfaceOpticDiscAnnotation(polygon=np.array([[1, 2, 3], [4, 5, 6]]))

    def test_from_ellipse(self):
        """Test creation from ellipse parameters."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            rotation=np.pi/4,
            shape=(100, 100),
            num_points=50
        )

        assert od.polygon.shape == (50, 2)
        assert od.shape == (100, 100)

        # Check that center is approximately correct
        center = od.center
        assert abs(center[0] - 50) < 2  # Allow small deviation
        assert abs(center[1] - 50) < 2

    def test_from_ellipse_default_rotation(self):
        """Test ellipse creation with default rotation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        assert od.polygon.shape[0] == 64  # default num_points
        assert od.shape == (100, 100)

    def test_from_mask(self):
        """Test creation from binary mask."""
        # Create a simple square mask
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True

        od = EyeEnfaceOpticDiscAnnotation.from_mask(mask)

        assert od.shape == (60, 60)
        assert od.polygon.shape[0] > 0
        assert od.polygon.shape[1] == 2

    def test_from_mask_no_contours(self):
        """Test that error is raised when mask has no contours."""
        mask = np.zeros((60, 60), dtype=bool)

        with pytest.raises(ValueError, match='No contours found'):
            EyeEnfaceOpticDiscAnnotation.from_mask(mask)

    def test_from_mask_multiple_contours(self):
        """Test that largest contour is selected when multiple exist."""
        mask = np.zeros((100, 100), dtype=bool)
        # Small contour
        mask[10:20, 10:20] = True
        # Large contour
        mask[30:70, 30:70] = True

        od = EyeEnfaceOpticDiscAnnotation.from_mask(mask)

        # Should use the larger contour
        center = od.center
        assert 30 < center[0] < 70
        assert 30 < center[1] < 70

    def test_polygon_property_returns_copy(self):
        """Test that polygon property returns a copy."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Get polygon and modify it
        polygon_copy = od.polygon
        polygon_copy[0, 0] = 999

        # Original should be unchanged
        assert od.polygon[0, 0] == 10

    def test_mask_property(self):
        """Test mask generation from polygon."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        mask = od.mask
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_mask_requires_shape(self):
        """Test that mask generation fails without shape."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon)

        with pytest.raises(ValueError, match='Shape must be set'):
            _ = od.mask

    def test_mask_caching(self):
        """Test that mask is cached."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        mask1 = od.mask
        mask2 = od.mask

        # Should return the same cached object
        assert mask1 is mask2

    def test_center_property(self):
        """Test center calculation."""
        # Create circular polygon
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 60),
            minor_axis=20,
            major_axis=20,
            shape=(100, 120)
        )

        center = od.center
        assert abs(center[0] - 50) < 2
        assert abs(center[1] - 60) < 2

    def test_width_property(self):
        """Test width calculation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        width = od.width
        # Should be approximately the minor axis
        assert 25 < width < 35

    def test_height_property(self):
        """Test height calculation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        height = od.height
        # Should be approximately the major axis
        # Should be approximately the minor_axis (vertical extent)
        assert 15 < height < 25

    def test_fit_ellipse_without_shape(self):
        """Test ellipse fitting when shape is not set."""
        # Create small polygon
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon)  # No shape

        # Should still be able to fit ellipse
        center = od.center
        width = od.width
        height = od.height

        assert center[0] > 0
        assert center[1] > 0
        assert width > 0
        assert height > 0

    def test_ellipse_caching(self):
        """Test that fitted ellipse is cached."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Access properties that trigger fitting
        center1 = od.center
        width1 = od.width
        height1 = od.height

        # Access again
        center2 = od.center
        width2 = od.width
        height2 = od.height

        # Should be identical (cached)
        assert center1 == center2
        assert width1 == width2
        assert height1 == height2

    def test_round_trip_ellipse_to_mask_to_polygon(self):
        """Test creating from ellipse, getting mask, and recreating."""
        # Create from ellipse
        od1 = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=30,
            major_axis=40,
            shape=(100, 100)
        )

        # Get mask and recreate
        mask = od1.mask
        od2 = EyeEnfaceOpticDiscAnnotation.from_mask(mask)

        # Centers should be similar
        center1 = od1.center
        center2 = od2.center
        assert abs(center1[0] - center2[0]) < 5
        assert abs(center1[1] - center2[1]) < 5

    def test_scale_transformation(self):
        """Test scaling transformation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        # Scale up
        od_large = od.scale(2.0)
        assert abs(od_large.width - od.width * 2) < 2
        assert abs(od_large.height - od.height * 2) < 2
        assert abs(od_large.center[0] - od.center[0]) < 1
        assert abs(od_large.center[1] - od.center[1]) < 1

        # Scale down
        od_small = od.scale(0.5)
        assert abs(od_small.width - od.width * 0.5) < 2
        assert abs(od_small.height - od.height * 0.5) < 2

        # Original should be unchanged
        # width measures horizontal extent (major_axis), height measures vertical extent (minor_axis)
        assert abs(od.width - 30) < 2  # horizontal = major_axis
        assert abs(od.height - 20) < 2  # vertical = minor_axis

    def test_scale_with_custom_center(self):
        """Test scaling with custom center point."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Scale around origin
        od_scaled = od.scale(2.0, center=(0, 0))

        # Vertices should be doubled
        expected = polygon * 2.0
        np.testing.assert_array_almost_equal(od_scaled.polygon, expected, decimal=1)

    def test_translate_transformation(self):
        """Test translation transformation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        # Translate
        od_moved = od.translate(10, 20)

        # Center should move by translation
        assert abs(od_moved.center[0] - (od.center[0] + 10)) < 1
        assert abs(od_moved.center[1] - (od.center[1] + 20)) < 1

        # Size should be unchanged
        assert abs(od_moved.width - od.width) < 1
        assert abs(od_moved.height - od.height) < 1

        # Original should be unchanged
        assert abs(od.center[0] - 50) < 1

    def test_rotate_transformation(self):
        """Test rotation transformation."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        # Rotate 90 degrees
        od_rotated = od.rotate(np.pi / 2)

        # Center should stay approximately the same
        assert abs(od_rotated.center[0] - od.center[0]) < 2
        assert abs(od_rotated.center[1] - od.center[1]) < 2

        # After 90-degree rotation, width and height should swap
        # (since width/height are now horizontal/vertical extents, not ellipse axes)
        assert abs(od_rotated.width - od.height) < 5, 'After 90° rotation, width should match original height'
        assert abs(od_rotated.height - od.width) < 5, 'After 90° rotation, height should match original width'

        # Original should be unchanged
        assert abs(od.center[0] - 50) < 1

    def test_rotate_with_custom_center(self):
        """Test rotation with custom center point."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Rotate 180 degrees around origin
        od_rotated = od.rotate(np.pi, center=(0, 0))

        # Points should be negated
        expected = -polygon
        np.testing.assert_array_almost_equal(od_rotated.polygon, expected, decimal=1)

    def test_transform_2x2_matrix(self):
        """Test affine transformation with 2x2 matrix."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Scale matrix
        scale_matrix = np.array([[2, 0], [0, 2]], dtype=np.float64)
        od_transformed = od.transform(scale_matrix, center=(0, 0))

        # Should double the coordinates
        expected = polygon * 2.0
        np.testing.assert_array_almost_equal(od_transformed.polygon, expected, decimal=1)

    def test_transform_2x3_matrix(self):
        """Test affine transformation with 2x3 matrix (includes translation)."""
        polygon = np.array([[10, 10], [10, 30], [30, 30], [30, 10]], dtype=np.float64)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Scale and translate: [x', y'] = [x, y] @ [[2, 0], [0, 2]] + [5, 10]
        # In our (y, x) format, we need to swap
        matrix = np.array([[2, 0, 5], [0, 2, 10]], dtype=np.float64)
        od_transformed = od.transform(matrix)

        # Should scale and translate
        # x' = 2*x + 5, y' = 2*y + 10
        expected = polygon * 2 + np.array([5, 10])
        print(f'DEBUG: polygon={polygon}')
        print(f'DEBUG: expected={expected}')
        print(f'DEBUG: actual={od_transformed.polygon}')

        np.testing.assert_array_almost_equal(od_transformed.polygon, expected, decimal=1)
        od = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=(50, 50))

        # Wrong shape
        with pytest.raises(ValueError, match='Matrix must be 2x2 or 2x3'):
            od.transform(np.array([[1, 2, 3, 4]]))

    def test_transformations_are_immutable(self):
        """Test that transformations don't modify the original object."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        original_center = od.center
        original_polygon = od.polygon.copy()

        # Apply various transformations
        _ = od.scale(2.0)
        _ = od.translate(10, 20)
        _ = od.rotate(np.pi / 4)
        _ = od.transform(np.array([[1, 0.5], [0, 1]]))

        # Original should be completely unchanged
        assert od.center == original_center
        np.testing.assert_array_equal(od.polygon, original_polygon)

    def test_chained_transformations(self):
        """Test chaining multiple transformations."""
        od = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 50),
            minor_axis=20,
            major_axis=30,
            shape=(100, 100)
        )

        # Chain transformations
        od_transformed = (od
            .scale(1.5)
            .rotate(np.pi / 6)
            .translate(10, 20))

        # Should be a valid EyeEnfaceOpticDiscAnnotation
        assert isinstance(od_transformed, EyeEnfaceOpticDiscAnnotation)
        assert od_transformed.shape == (100, 100)

        # Original should be unchanged
        assert abs(od.center[0] - 50) < 1
        assert abs(od.center[1] - 50) < 1
