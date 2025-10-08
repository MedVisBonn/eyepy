"""Test that EyeEnface transformations update metadata correctly."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeEnfaceMeta


@pytest.fixture
def mock_enface():
    """Create a mock EyeEnface instance for testing."""
    # Create a simple test image
    data = np.random.rand(100, 100).astype(np.float32)
    
    # Create metadata with scale information
    meta = EyeEnfaceMeta(
        scale_x=10.0,  # 10 µm/pixel
        scale_y=10.0,  # 10 µm/pixel
        scale_unit='µm'
    )
    
    return EyeEnface(data=data, meta=meta)


def test_scale_updates_metadata(mock_enface):
    """Test that scaling updates scale_x and scale_y in metadata."""
    # Scale by 2x in both dimensions
    scaled = mock_enface.scale(scale_y=2.0, scale_x=2.0)
    
    # After scaling 2x, each pixel represents half the physical distance
    assert scaled.scale_x == pytest.approx(5.0)
    assert scaled.scale_y == pytest.approx(5.0)
    assert scaled.meta['scale_unit'] == 'µm'
    
    # Original should be unchanged
    assert mock_enface.scale_x == pytest.approx(10.0)
    assert mock_enface.scale_y == pytest.approx(10.0)


def test_scale_down_updates_metadata(mock_enface):
    """Test that scaling down updates metadata correctly."""
    # Scale by 0.5x in both dimensions (shrink)
    scaled = mock_enface.scale(scale_y=0.5, scale_x=0.5)
    
    # After scaling 0.5x, each pixel represents twice the physical distance
    assert scaled.scale_x == pytest.approx(20.0)
    assert scaled.scale_y == pytest.approx(20.0)


def test_scale_anisotropic_updates_metadata(mock_enface):
    """Test that anisotropic scaling updates metadata correctly."""
    # Scale by different factors in x and y
    scaled = mock_enface.scale(scale_y=2.0, scale_x=3.0)
    
    # Check each dimension independently
    assert scaled.scale_x == pytest.approx(10.0 / 3.0)
    assert scaled.scale_y == pytest.approx(10.0 / 2.0)


def test_translate_preserves_metadata(mock_enface):
    """Test that translation doesn't change scale metadata."""
    translated = mock_enface.translate(offset_y=10, offset_x=20)
    
    # Translation should not affect scale
    assert translated.scale_x == pytest.approx(10.0)
    assert translated.scale_y == pytest.approx(10.0)


def test_rotate_preserves_metadata(mock_enface):
    """Test that rotation doesn't change scale metadata."""
    rotated = mock_enface.rotate(angle=45)
    
    # Pure rotation should not affect scale
    assert rotated.scale_x == pytest.approx(10.0)
    assert rotated.scale_y == pytest.approx(10.0)


def test_transform_with_scaling_updates_metadata(mock_enface):
    """Test that a custom transformation matrix with scaling updates metadata."""
    # Create a transformation matrix with 2x scaling in both dimensions
    matrix = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    transformed = mock_enface.transform(matrix, output_shape=(200, 200))
    
    # Check metadata was updated
    assert transformed.scale_x == pytest.approx(5.0)
    assert transformed.scale_y == pytest.approx(5.0)


def test_transform_with_rotation_and_scaling(mock_enface):
    """Test that combined rotation and scaling updates metadata correctly."""
    # Create a transformation matrix with 45° rotation and 2x scaling
    angle = np.deg2rad(45)
    scale_factor = 2.0
    matrix = np.array([
        [scale_factor * np.cos(angle), -scale_factor * np.sin(angle), 0.0],
        [scale_factor * np.sin(angle), scale_factor * np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    transformed = mock_enface.transform(matrix, output_shape=(200, 200))
    
    # The scaling factor should be extracted correctly despite rotation
    assert transformed.scale_x == pytest.approx(5.0, rel=1e-5)
    assert transformed.scale_y == pytest.approx(5.0, rel=1e-5)


def test_chained_transformations_update_metadata(mock_enface):
    """Test that chained transformations correctly update metadata."""
    # Scale by 2x
    scaled_once = mock_enface.scale(scale_y=2.0, scale_x=2.0)
    assert scaled_once.scale_x == pytest.approx(5.0)
    
    # Scale by 2x again
    scaled_twice = scaled_once.scale(scale_y=2.0, scale_x=2.0)
    assert scaled_twice.scale_x == pytest.approx(2.5)
    
    # Original should still be unchanged
    assert mock_enface.scale_x == pytest.approx(10.0)


def test_metadata_copy_is_independent(mock_enface):
    """Test that metadata is properly copied and modifications don't affect original."""
    scaled = mock_enface.scale(scale_y=2.0, scale_x=2.0)
    
    # Modify the transformed metadata
    scaled.meta['custom_field'] = 'test_value'
    
    # Original metadata should not be affected
    assert 'custom_field' not in mock_enface.meta
    
    # Original scale should not be affected
    assert mock_enface.scale_x == pytest.approx(10.0)
    assert mock_enface.scale_y == pytest.approx(10.0)
