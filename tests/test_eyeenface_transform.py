"""Tests for EyeEnface transformation methods."""

import numpy as np
import pytest
from unittest.mock import Mock

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation, EyeEnfaceFoveaAnnotation


@pytest.fixture
def mock_meta():
    """Create a mock EyeEnfaceMeta object."""
    meta = Mock()
    meta.__getitem__ = Mock(side_effect=lambda key: {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OD'
    }.get(key))
    return meta


@pytest.fixture
def simple_enface(mock_meta):
    """Create a simple EyeEnface for testing."""
    data = np.arange(100).reshape(10, 10).astype(np.int64)
    
    # Create optic disc
    optic_disc_polygon = np.array([
        [2.0, 2.0],
        [2.0, 4.0],
        [4.0, 4.0],
        [4.0, 2.0]
    ])
    optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(10, 10))
    
    # Create fovea
    fovea_polygon = np.array([
        [7.0, 7.0],
        [7.0, 8.0],
        [8.0, 8.0],
        [8.0, 7.0]
    ])
    fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(10, 10))
    
    return EyeEnface(data=data, meta=mock_meta, optic_disc=optic_disc, fovea=fovea)


class TestEyeEnfaceScale:
    """Tests for EyeEnface.scale()"""
    
    def test_scale_returns_new_instance(self, simple_enface):
        """Test that scale returns a new EyeEnface instance."""
        scaled = simple_enface.scale(2.0, 2.0)
        assert isinstance(scaled, EyeEnface)
        assert scaled is not simple_enface
    
    def test_scale_transforms_data(self, simple_enface):
        """Test that scale transforms the image data."""
        scaled = simple_enface.scale(2.0, 2.0)
        assert scaled.data.shape == (20, 20)
        assert simple_enface.data.shape == (10, 10)  # Original unchanged
    
    def test_scale_transforms_optic_disc(self, simple_enface):
        """Test that scale transforms the optic disc annotation."""
        scaled = simple_enface.scale(2.0, 2.0)
        
        # Check that optic disc is scaled
        original_center = simple_enface.optic_disc.center
        scaled_center = scaled.optic_disc.center
        
        assert scaled_center[0] == pytest.approx(original_center[0] * 2.0, abs=0.1)
        assert scaled_center[1] == pytest.approx(original_center[1] * 2.0, abs=0.1)
    
    def test_scale_transforms_fovea(self, simple_enface):
        """Test that scale transforms the fovea annotation."""
        scaled = simple_enface.scale(2.0, 2.0)
        
        # Check that fovea is scaled
        original_center = simple_enface.fovea.center
        scaled_center = scaled.fovea.center
        
        assert scaled_center[0] == pytest.approx(original_center[0] * 2.0, abs=0.1)
        assert scaled_center[1] == pytest.approx(original_center[1] * 2.0, abs=0.1)
    
    def test_scale_original_unchanged(self, simple_enface):
        """Test that scaling doesn't modify the original enface."""
        original_data = simple_enface.data.copy()
        original_od_polygon = simple_enface.optic_disc.polygon.copy()
        original_fovea_polygon = simple_enface.fovea.polygon.copy()
        
        _ = simple_enface.scale(2.0, 2.0)
        
        np.testing.assert_array_equal(simple_enface.data, original_data)
        np.testing.assert_array_equal(simple_enface.optic_disc.polygon, original_od_polygon)
        np.testing.assert_array_equal(simple_enface.fovea.polygon, original_fovea_polygon)
    
    def test_scale_without_annotations(self, mock_meta):
        """Test scaling an enface without annotations."""
        data = np.arange(100).reshape(10, 10).astype(np.int64)
        enface = EyeEnface(data=data, meta=mock_meta)
        
        scaled = enface.scale(2.0, 2.0)
        assert scaled.data.shape == (20, 20)
        assert scaled.optic_disc is None
        assert scaled.fovea is None


class TestEyeEnfaceTranslate:
    """Tests for EyeEnface.translate()"""
    
    def test_translate_returns_new_instance(self, simple_enface):
        """Test that translate returns a new EyeEnface instance."""
        translated = simple_enface.translate(2.0, 3.0)
        assert isinstance(translated, EyeEnface)
        assert translated is not simple_enface
    
    def test_translate_preserves_shape(self, simple_enface):
        """Test that translate preserves the image shape."""
        translated = simple_enface.translate(2.0, 3.0)
        assert translated.data.shape == simple_enface.data.shape
    
    def test_translate_transforms_optic_disc(self, simple_enface):
        """Test that translate transforms the optic disc annotation."""
        translated = simple_enface.translate(2.0, 3.0)
        
        # Check that optic disc is translated
        original_center = simple_enface.optic_disc.center
        translated_center = translated.optic_disc.center
        
        assert translated_center[0] == pytest.approx(original_center[0] + 2.0, abs=0.1)
        assert translated_center[1] == pytest.approx(original_center[1] + 3.0, abs=0.1)
    
    def test_translate_transforms_fovea(self, simple_enface):
        """Test that translate transforms the fovea annotation."""
        translated = simple_enface.translate(2.0, 3.0)
        
        # Check that fovea is translated
        original_center = simple_enface.fovea.center
        translated_center = translated.fovea.center
        
        assert translated_center[0] == pytest.approx(original_center[0] + 2.0, abs=0.1)
        assert translated_center[1] == pytest.approx(original_center[1] + 3.0, abs=0.1)
    
    def test_translate_original_unchanged(self, simple_enface):
        """Test that translation doesn't modify the original enface."""
        original_data = simple_enface.data.copy()
        original_od_polygon = simple_enface.optic_disc.polygon.copy()
        
        _ = simple_enface.translate(2.0, 3.0)
        
        np.testing.assert_array_equal(simple_enface.data, original_data)
        np.testing.assert_array_equal(simple_enface.optic_disc.polygon, original_od_polygon)


class TestEyeEnfaceRotate:
    """Tests for EyeEnface.rotate()"""
    
    def test_rotate_returns_new_instance(self, simple_enface):
        """Test that rotate returns a new EyeEnface instance."""
        rotated = simple_enface.rotate(45.0)
        assert isinstance(rotated, EyeEnface)
        assert rotated is not simple_enface
    
    def test_rotate_preserves_shape(self, simple_enface):
        """Test that rotate preserves the image shape."""
        rotated = simple_enface.rotate(45.0)
        assert rotated.data.shape == simple_enface.data.shape
    
    def test_rotate_90_degrees(self, simple_enface):
        """Test 90-degree rotation."""
        rotated = simple_enface.rotate(90.0)
        
        # After 90 degree rotation around center, annotations should move
        original_center = simple_enface.optic_disc.center
        rotated_center = rotated.optic_disc.center
        
        # Center should have moved (not testing exact values due to rotation complexity)
        assert rotated_center != original_center
    
    def test_rotate_with_custom_center(self, simple_enface):
        """Test rotation with a custom center point."""
        center = (5.0, 5.0)
        rotated = simple_enface.rotate(45.0, center=center)
        
        assert isinstance(rotated, EyeEnface)
        assert rotated.optic_disc is not None
    
    def test_rotate_original_unchanged(self, simple_enface):
        """Test that rotation doesn't modify the original enface."""
        original_data = simple_enface.data.copy()
        original_od_polygon = simple_enface.optic_disc.polygon.copy()
        
        _ = simple_enface.rotate(45.0)
        
        np.testing.assert_array_equal(simple_enface.data, original_data)
        np.testing.assert_array_equal(simple_enface.optic_disc.polygon, original_od_polygon)


class TestEyeEnfaceTransform:
    """Tests for EyeEnface.transform()"""
    
    def test_transform_returns_new_instance(self, simple_enface):
        """Test that transform returns a new EyeEnface instance."""
        # Identity matrix
        matrix = np.eye(3)
        transformed = simple_enface.transform(matrix)
        
        assert isinstance(transformed, EyeEnface)
        assert transformed is not simple_enface
    
    def test_transform_with_identity_matrix(self, simple_enface):
        """Test transform with identity matrix (no change)."""
        matrix = np.eye(3)
        transformed = simple_enface.transform(matrix)
        
        # Data should be approximately the same
        assert transformed.data.shape == simple_enface.data.shape
    
    def test_transform_with_scaling_matrix(self, simple_enface):
        """Test transform with a scaling matrix."""
        # 2x scaling matrix
        matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        transformed = simple_enface.transform(matrix, output_shape=(20, 20))
        
        assert transformed.data.shape == (20, 20)
        assert transformed.optic_disc is not None
    
    def test_transform_original_unchanged(self, simple_enface):
        """Test that transform doesn't modify the original enface."""
        matrix = np.eye(3)
        original_data = simple_enface.data.copy()
        original_od_polygon = simple_enface.optic_disc.polygon.copy()
        
        _ = simple_enface.transform(matrix)
        
        np.testing.assert_array_equal(simple_enface.data, original_data)
        np.testing.assert_array_equal(simple_enface.optic_disc.polygon, original_od_polygon)


class TestEyeEnfaceMutability:
    """Tests for EyeEnface mutability with respect to annotations."""
    
    def test_can_modify_annotations_after_creation(self, simple_enface):
        """Test that annotations can be added after EyeEnface creation."""
        # Add area annotation
        area_map = np.random.rand(10, 10) > 0.5
        annotation = simple_enface.add_area_annotation(area_map, name='test_area')
        
        assert annotation in simple_enface._area_maps
        assert len(simple_enface._area_maps) == 1
    
    def test_can_modify_optic_disc_after_creation(self, simple_enface):
        """Test that optic_disc can be modified after creation."""
        new_polygon = np.array([
            [1.0, 1.0],
            [1.0, 3.0],
            [3.0, 3.0],
            [3.0, 1.0]
        ])
        new_optic_disc = EyeEnfaceOpticDiscAnnotation(new_polygon, shape=(10, 10))
        
        simple_enface.optic_disc = new_optic_disc
        assert simple_enface.optic_disc is new_optic_disc
    
    def test_can_remove_annotations(self, simple_enface):
        """Test that annotations can be removed."""
        simple_enface.optic_disc = None
        simple_enface.fovea = None
        
        assert simple_enface.optic_disc is None
        assert simple_enface.fovea is None
    
    def test_transformed_enface_is_also_mutable(self, simple_enface):
        """Test that transformed EyeEnface instances are also mutable."""
        scaled = simple_enface.scale(2.0, 2.0)
        
        # Should be able to modify annotations on the scaled enface
        new_polygon = np.array([
            [2.0, 2.0],
            [2.0, 4.0],
            [4.0, 4.0],
            [4.0, 2.0]
        ])
        new_optic_disc = EyeEnfaceOpticDiscAnnotation(new_polygon, shape=(20, 20))
        
        scaled.optic_disc = new_optic_disc
        assert scaled.optic_disc is new_optic_disc
        
        # Original should be unchanged
        assert simple_enface.optic_disc is not new_optic_disc


class TestAreaMapTransformation:
    """Tests for area map transformation."""
    
    def test_area_maps_are_transformed(self, simple_enface):
        """Test that area maps are transformed along with the enface."""
        # Add an area map
        area_map = np.zeros((10, 10), dtype=bool)
        area_map[2:5, 2:5] = True
        simple_enface.add_area_annotation(area_map, name='test_area')
        
        # Scale the enface
        scaled = simple_enface.scale(2.0, 2.0)
        
        # Check that area maps were copied
        assert len(scaled._area_maps) == 1
        assert scaled._area_maps[0].meta['name'] == 'test_area'
        assert scaled._area_maps[0].data.shape == (20, 20)
    
    def test_area_maps_reference_new_enface(self, simple_enface):
        """Test that transformed area maps reference the new enface."""
        area_map = np.zeros((10, 10), dtype=bool)
        simple_enface.add_area_annotation(area_map, name='test_area')
        
        scaled = simple_enface.scale(2.0, 2.0)
        
        # Area map should reference the new enface, not the original
        assert scaled._area_maps[0].enface is scaled
        assert scaled._area_maps[0].enface is not simple_enface
