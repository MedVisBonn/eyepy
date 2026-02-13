"""Tests for EyeEnface laterality validation."""

import numpy as np
import pytest

from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeEnfaceMeta


@pytest.fixture
def mock_meta_od():
    """Create an EyeEnfaceMeta for right eye (OD)."""
    return EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm', laterality='OD')


@pytest.fixture
def mock_meta_os():
    """Create an EyeEnfaceMeta for left eye (OS)."""
    return EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm', laterality='OS')


@pytest.fixture
def mock_meta_no_laterality():
    """Create an EyeEnfaceMeta without laterality info."""
    return EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm')


class TestLateralityValidation:
    """Tests for laterality validation in EyeEnface."""

    def test_right_eye_correct_anatomy(self, mock_meta_od):
        """Test that right eye (OD) with optic disc to the right of fovea is
        valid."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the right (higher x) of fovea - correct for OD
        optic_disc_polygon = np.array([
            [40.0, 70.0],
            [40.0, 80.0],
            [50.0, 80.0],
            [50.0, 70.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 20.0],
            [45.0, 30.0],
            [55.0, 30.0],
            [55.0, 20.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should not raise
        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)
        # _validate_laterality doesn't return a value, so we just check it doesn't raise
        enface._validate_laterality()

    def test_right_eye_incorrect_anatomy(self, mock_meta_od):
        """Test that right eye (OD) with optic disc to the left of fovea raises
        error."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the left (lower x) of fovea - incorrect for OD
        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should raise ValueError
        with pytest.raises(ValueError, match='Laterality mismatch.*Right eye'):
            EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)

    def test_left_eye_correct_anatomy(self, mock_meta_os):
        """Test that left eye (OS) with optic disc to the left of fovea is
        valid."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the left (lower x) of fovea - correct for OS
        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should not raise
        enface = EyeEnface(data=data, meta=mock_meta_os, optic_disc=optic_disc, fovea=fovea)
        # _validate_laterality doesn't return a value, so we just check it doesn't raise
        enface._validate_laterality()

    def test_left_eye_incorrect_anatomy(self, mock_meta_os):
        """Test that left eye (OS) with optic disc to the right of fovea raises
        error."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the right (higher x) of fovea - incorrect for OS
        optic_disc_polygon = np.array([
            [40.0, 70.0],
            [40.0, 80.0],
            [50.0, 80.0],
            [50.0, 70.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 20.0],
            [45.0, 30.0],
            [55.0, 30.0],
            [55.0, 20.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should raise ValueError
        with pytest.raises(ValueError, match='Laterality mismatch.*Left eye'):
            EyeEnface(data=data, meta=mock_meta_os, optic_disc=optic_disc, fovea=fovea)

    def test_no_laterality_info_infers_laterality(self, mock_meta_no_laterality):
        """Test that missing laterality info is inferred from anatomy."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the left of fovea -> should infer OS (left eye)
        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should not raise and should infer laterality as OS
        enface = EyeEnface(data=data, meta=mock_meta_no_laterality, optic_disc=optic_disc, fovea=fovea)
        # _validate_laterality doesn't return a value, so we just check it doesn't raise
        enface._validate_laterality()
        # Verify laterality was inferred
        assert enface.laterality == 'OS'

    def test_only_optic_disc_no_validation(self, mock_meta_od):
        """Test that having only optic disc doesn't trigger validation."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        # Should not raise
        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=None)
        # _validate_laterality doesn't return a value, so we just check it doesn't raise
        enface._validate_laterality()

    def test_only_fovea_no_validation(self, mock_meta_od):
        """Test that having only fovea doesn't trigger validation."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should not raise
        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=None, fovea=fovea)
        # _validate_laterality doesn't return a value, so we just check it doesn't raise
        enface._validate_laterality()

    def test_right_laterality_variants(self, mock_meta_od):
        """Test that different right eye indicators work (OD, R, RIGHT)."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the right (higher x) of fovea - correct for OD
        optic_disc_polygon = np.array([
            [40.0, 70.0],
            [40.0, 80.0],
            [50.0, 80.0],
            [50.0, 70.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 20.0],
            [45.0, 30.0],
            [55.0, 30.0],
            [55.0, 20.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Test all variants
        for laterality in ['OD', 'R', 'RIGHT', 'od', 'r', 'right']:
            meta = EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm', laterality=laterality)

            # Should not raise for any variant
            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            # _validate_laterality doesn't return a value, so we just check it doesn't raise
            enface._validate_laterality()

    def test_left_laterality_variants(self, mock_meta_os):
        """Test that different left eye indicators work (OS, L, LEFT)."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Optic disc to the left (lower x) of fovea - correct for OS
        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Test all variants
        for laterality in ['OS', 'L', 'LEFT', 'os', 'l', 'left']:
            meta = EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm', laterality=laterality)

            # Should not raise for any variant
            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            # _validate_laterality doesn't return a value, so we just check it doesn't raise
            enface._validate_laterality()
