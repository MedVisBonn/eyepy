"""Tests for EyeEnface laterality validation."""

import numpy as np
import pytest
from unittest.mock import Mock

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation, EyeEnfaceFoveaAnnotation


@pytest.fixture
def mock_meta_od():
    """Create a mock EyeEnfaceMeta for right eye (OD)."""
    meta = Mock()
    meta.__getitem__ = Mock(side_effect=lambda key: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OD'
    }.get(key))
    meta.get = Mock(side_effect=lambda key, default=None: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OD'
    }.get(key, default))
    return meta


@pytest.fixture
def mock_meta_os():
    """Create a mock EyeEnfaceMeta for left eye (OS)."""
    meta = Mock()
    meta.__getitem__ = Mock(side_effect=lambda key: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OS'
    }.get(key))
    meta.get = Mock(side_effect=lambda key, default=None: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OS'
    }.get(key, default))
    return meta


@pytest.fixture
def mock_meta_no_laterality():
    """Create a mock EyeEnfaceMeta without laterality info."""
    meta = Mock()
    meta.__getitem__ = Mock(side_effect=lambda key: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm'
    }.get(key))
    meta.get = Mock(side_effect=lambda key, default=None: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm'
    }.get(key, default))
    return meta


class TestLateralityValidation:
    """Tests for laterality validation in EyeEnface."""
    
    def test_right_eye_correct_anatomy(self, mock_meta_od):
        """Test that right eye (OD) with optic disc to the right of fovea is valid."""
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
        assert enface.validate_laterality() is True
    
    def test_right_eye_incorrect_anatomy(self, mock_meta_od):
        """Test that right eye (OD) with optic disc to the left of fovea raises error."""
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
        with pytest.raises(ValueError, match="Laterality mismatch.*Right eye"):
            EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)
    
    def test_left_eye_correct_anatomy(self, mock_meta_os):
        """Test that left eye (OS) with optic disc to the left of fovea is valid."""
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
        assert enface.validate_laterality() is True
    
    def test_left_eye_incorrect_anatomy(self, mock_meta_os):
        """Test that left eye (OS) with optic disc to the right of fovea raises error."""
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
        with pytest.raises(ValueError, match="Laterality mismatch.*Left eye"):
            EyeEnface(data=data, meta=mock_meta_os, optic_disc=optic_disc, fovea=fovea)
    
    def test_no_laterality_info_no_validation(self, mock_meta_no_laterality):
        """Test that missing laterality info doesn't trigger validation."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)
        
        # Any positions - should not validate without laterality
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
        enface = EyeEnface(data=data, meta=mock_meta_no_laterality, optic_disc=optic_disc, fovea=fovea)
        assert enface.validate_laterality() is True
    
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
        assert enface.validate_laterality() is True
    
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
        assert enface.validate_laterality() is True
    
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
            meta = Mock()
            meta.__getitem__ = Mock(side_effect=lambda key: {
                'scale_x': 1.0,
                'scale_y': 1.0,
                'scale_unit': 'mm',
                'laterality': laterality
            }.get(key))
            meta.get = Mock(side_effect=lambda key, default=None: {
                'scale_x': 1.0,
                'scale_y': 1.0,
                'scale_unit': 'mm',
                'laterality': laterality
            }.get(key, default))
            
            # Should not raise for any variant
            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            assert enface.validate_laterality() is True
    
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
            meta = Mock()
            meta.__getitem__ = Mock(side_effect=lambda key: {
                'scale_x': 1.0,
                'scale_y': 1.0,
                'scale_unit': 'mm',
                'laterality': laterality
            }.get(key))
            meta.get = Mock(side_effect=lambda key, default=None: {
                'scale_x': 1.0,
                'scale_y': 1.0,
                'scale_unit': 'mm',
                'laterality': laterality
            }.get(key, default))
            
            # Should not raise for any variant
            enface = EyeEnface(data=data, meta=meta, optic_disc=optic_disc, fovea=fovea)
            assert enface.validate_laterality() is True
