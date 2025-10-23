"""Tests for EyeEnface standard_rotation method."""

import numpy as np
import pytest

from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeEnfaceMeta


@pytest.fixture
def enface_with_annotations():
    """Create an EyeEnface with optic disc and fovea for a right eye (OD)."""
    data = np.random.rand(100, 100).astype(np.float32)
    meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality='OD')

    # For right eye (OD): optic disc should end up on the right, fovea on the left
    # Create optic disc at (row=50, col=60) - to the right
    optic_disc_polygon = np.array([
        [48.0, 58.0],
        [48.0, 62.0],
        [52.0, 62.0],
        [52.0, 58.0]
    ])
    optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

    # Create fovea at (row=50, col=40) - to the left
    fovea_polygon = np.array([
        [50.0, 40.0]
    ])
    fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

    return EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)


@pytest.fixture
def enface_horizontal_alignment():
    """Create an EyeEnface with optic disc and fovea already horizontally aligned
    (right eye)."""
    data = np.random.rand(100, 100).astype(np.float32)
    meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality='OD')

    # Right eye: optic disc on right (col=70), fovea on left (col=30), same horizontal line (row=50)
    optic_disc_polygon = np.array([
        [48.0, 68.0],
        [48.0, 72.0],
        [52.0, 72.0],
        [52.0, 68.0]
    ])
    optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

    fovea_polygon = np.array([
        [50.0, 40.0]
    ])
    fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

    return EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)


@pytest.fixture
def enface_left_eye():
    """Create an EyeEnface for a left eye (OS)."""
    data = np.random.rand(100, 100).astype(np.float32)
    meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality='OS')

    # For left eye (OS): optic disc should end up on the left, fovea on the right
    # Create optic disc at (row=50, col=40) - to the left
    optic_disc_polygon = np.array([
        [48.0, 38.0],
        [48.0, 42.0],
        [52.0, 42.0],
        [52.0, 38.0]
    ])
    optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

    # Create fovea at (row=50, col=60) - to the right
    fovea_polygon = np.array([
        [48.0, 58.0],
        [48.0, 62.0],
        [52.0, 62.0],
        [52.0, 58.0]
    ])
    fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

    return EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)


class TestStandardRotation:
    """Test suite for the standard_rotation method."""

    def test_standard_rotation_returns_new_instance(self, enface_with_annotations):
        """Test that standard_rotation returns a new EyeEnface instance."""
        rotated = enface_with_annotations.standard_rotation()

        assert rotated is not enface_with_annotations
        assert isinstance(rotated, EyeEnface)

    def test_standard_rotation_aligns_centers_horizontally(self, enface_with_annotations):
        """Test that centers are aligned horizontally after rotation."""
        rotated = enface_with_annotations.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # Check that row-coordinates are approximately equal (within tolerance for floating point)
        # center is (row, col), so index 0 is row (vertical position)
        assert abs(od_center[0] - fovea_center[0]) < 0.5, \
            f'Centers not aligned: OD row={od_center[0]}, Fovea row={fovea_center[0]}'

    def test_standard_rotation_preserves_shape(self, enface_with_annotations):
        """Test that standard_rotation preserves image shape."""
        rotated = enface_with_annotations.standard_rotation()

        assert rotated.shape == enface_with_annotations.shape

    def test_standard_rotation_requires_optic_disc(self):
        """Test that standard_rotation raises error without optic disc."""
        data = np.random.rand(100, 100).astype(np.float32)
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')

        fovea_polygon = np.array([[48.0, 48.0], [48.0, 52.0], [52.0, 52.0], [52.0, 48.0]])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=meta, optic_disc=None, fovea=fovea)

        with pytest.raises(ValueError, match='standard_rotation requires optic_disc'):
            enface.standard_rotation()

    def test_standard_rotation_requires_fovea(self):
        """Test that standard_rotation raises error without fovea."""
        data = np.random.rand(100, 100).astype(np.float32)
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')

        optic_disc_polygon = np.array([[48.0, 48.0], [48.0, 52.0], [52.0, 52.0], [52.0, 48.0]])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=None)

        with pytest.raises(ValueError, match='standard_rotation requires fovea'):
            enface.standard_rotation()

    def test_standard_rotation_already_aligned(self, enface_horizontal_alignment):
        """Test standard_rotation on already aligned image (no rotation
        needed)."""
        rotated = enface_horizontal_alignment.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # Should still be aligned
        assert abs(od_center[0] - fovea_center[0]) < 0.5

    def test_standard_rotation_original_unchanged(self, enface_with_annotations):
        """Test that standard_rotation doesn't modify the original enface."""
        # Centers are tuples, need to convert to array first
        original_od_center = np.array(enface_with_annotations.optic_disc.center)
        original_fovea_center = np.array(enface_with_annotations.fovea.center)
        original_data = enface_with_annotations.data.copy()

        _ = enface_with_annotations.standard_rotation()

        # Original should be unchanged
        np.testing.assert_array_equal(enface_with_annotations.data, original_data)
        np.testing.assert_array_almost_equal(
            enface_with_annotations.optic_disc.center, original_od_center
        )
        np.testing.assert_array_almost_equal(
            enface_with_annotations.fovea.center, original_fovea_center
        )

    def test_standard_rotation_rotated_from_diagonal(self):
        """Test standard_rotation with diagonal initial positioning (right
        eye)."""
        data = np.random.rand(100, 100).astype(np.float32)
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality='OD')

        # Create annotations at 45-degree angle
        # OD at (row=60, col=60), Fovea at (row=40, col=40) - diagonal, fovea upper-left
        optic_disc_polygon = np.array([
            [58.0, 58.0],
            [58.0, 62.0],
            [62.0, 62.0],
            [62.0, 58.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [38.0, 38.0],
            [38.0, 42.0],
            [42.0, 42.0],
            [42.0, 38.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
        rotated = enface.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # Should be aligned horizontally
        assert abs(od_center[0] - fovea_center[0]) < 0.5

        # For right eye, OD should be to the right
        assert od_center[1] > fovea_center[1]

    def test_standard_rotation_preserves_metadata(self, enface_with_annotations):
        """Test that standard_rotation preserves metadata (rotation doesn't change
        scale)."""
        rotated = enface_with_annotations.standard_rotation()

        # Rotation shouldn't change pixel spacing
        assert rotated.scale_x == pytest.approx(enface_with_annotations.scale_x)
        assert rotated.scale_y == pytest.approx(enface_with_annotations.scale_y)
        assert rotated.meta['scale_unit'] == enface_with_annotations.meta['scale_unit']

    def test_standard_rotation_with_interpolation_order(self, enface_with_annotations):
        """Test standard_rotation with different interpolation orders."""
        # Should work with different interpolation orders
        rotated_nearest = enface_with_annotations.standard_rotation(order=0)
        rotated_bilinear = enface_with_annotations.standard_rotation(order=1)
        rotated_bicubic = enface_with_annotations.standard_rotation(order=3)

        # All should align centers
        for rotated in [rotated_nearest, rotated_bilinear, rotated_bicubic]:
            od_center = rotated.optic_disc.center
            fovea_center = rotated.fovea.center
            assert abs(od_center[0] - fovea_center[0]) < 0.5

    def test_standard_rotation_is_mutable(self, enface_with_annotations):
        """Test that the rotated enface is still mutable."""
        rotated = enface_with_annotations.standard_rotation()

        # Should be able to modify annotations
        new_polygon = np.array([[10.0, 10.0], [10.0, 15.0], [15.0, 15.0], [15.0, 10.0]])
        new_fovea = EyeEnfaceFoveaAnnotation(new_polygon, shape=rotated.shape)

        # This should not raise an error
        rotated.fovea = new_fovea
        assert rotated.fovea is new_fovea

    def test_standard_rotation_right_eye_orientation(self, enface_with_annotations):
        """Test that right eye (OD) has optic disc to the right of fovea after
        rotation."""
        rotated = enface_with_annotations.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # For right eye, optic disc should be to the right (larger x coordinate)
        assert od_center[1] > fovea_center[1], \
            f'Right eye: OD should be right of fovea. OD x={od_center[1]}, Fovea x={fovea_center[1]}'

        # Should be horizontally aligned
        assert abs(od_center[0] - fovea_center[0]) < 0.5

    def test_standard_rotation_left_eye_orientation(self, enface_left_eye):
        """Test that left eye (OS) has optic disc to the left of fovea after
        rotation."""
        rotated = enface_left_eye.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # For left eye, optic disc should be to the left (smaller x coordinate)
        assert od_center[1] < fovea_center[1], \
            f'Left eye: OD should be left of fovea. OD x={od_center[1]}, Fovea x={fovea_center[1]}'

        # Should be horizontally aligned
        assert abs(od_center[0] - fovea_center[0]) < 0.5

    def test_standard_rotation_without_laterality(self):
        """Test standard_rotation without laterality info (defaults to fovea on
        right)."""
        data = np.random.rand(100, 100).astype(np.float32)
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')  # No laterality

        # Create annotations at 45-degree angle
        optic_disc_polygon = np.array([
            [38.0, 38.0],
            [38.0, 42.0],
            [42.0, 42.0],
            [42.0, 38.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [58.0, 58.0],
            [58.0, 62.0],
            [62.0, 62.0],
            [62.0, 58.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
        rotated = enface.standard_rotation()

        od_center = rotated.optic_disc.center
        fovea_center = rotated.fovea.center

        # Should be aligned horizontally
        assert abs(od_center[0] - fovea_center[0]) < 0.5

        # Default behavior: fovea to the right (positive x direction)
        assert fovea_center[1] > od_center[1], \
            'Without laterality, fovea should default to the right'

    def test_standard_rotation_laterality_variants(self):
        """Test that different laterality formats work correctly."""
        data = np.random.rand(100, 100).astype(np.float32)

        # Test various right eye formats
        for laterality in ['OD', 'R', 'RIGHT', 'right', 'od']:
            meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality=laterality)

            # For right eye: OD on right (x=60), fovea on left (x=40)
            optic_disc_polygon = np.array([[48.0, 58.0], [48.0, 62.0], [52.0, 62.0], [52.0, 58.0]])
            optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

            fovea_polygon = np.array([[48.0, 38.0], [48.0, 42.0], [52.0, 42.0], [52.0, 38.0]])
            fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            rotated = enface.standard_rotation()

            od_center = rotated.optic_disc.center
            fovea_center = rotated.fovea.center

            # For right eye, OD should be to the right
            assert od_center[1] > fovea_center[1], \
                f"Right eye format '{laterality}': OD should be right of fovea"

        # Test various left eye formats
        for laterality in ['OS', 'L', 'LEFT', 'left', 'os']:
            meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm', laterality=laterality)

            # For left eye: OD on left (x=40), fovea on right (x=60)
            optic_disc_polygon = np.array([[48.0, 38.0], [48.0, 42.0], [52.0, 42.0], [52.0, 38.0]])
            optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

            fovea_polygon = np.array([[48.0, 58.0], [48.0, 62.0], [52.0, 62.0], [52.0, 58.0]])
            fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            rotated = enface.standard_rotation()

            od_center = rotated.optic_disc.center
            fovea_center = rotated.fovea.center

            # For left eye, OD should be to the left
            assert od_center[1] < fovea_center[1], \
                f"Left eye format '{laterality}': OD should be left of fovea"
