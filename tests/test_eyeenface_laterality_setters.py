"""Tests for laterality inference and validation when using property setters."""

from unittest.mock import Mock

import numpy as np
import pytest

from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
from eyepy.core.eyeenface import EyeEnface


@pytest.fixture
def mock_meta_no_laterality():
    """Create a mock EyeEnfaceMeta without laterality info."""
    meta = Mock()
    data = {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm'
    }
    meta.__getitem__ = Mock(side_effect=lambda key: data.get(key))
    meta.__setitem__ = Mock(side_effect=lambda key, value: data.update({key: value}))
    meta.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return meta


@pytest.fixture
def mock_meta_od():
    """Create a mock EyeEnfaceMeta for right eye (OD)."""
    meta = Mock()
    data = {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OD'
    }
    meta.__getitem__ = Mock(side_effect=lambda key: data.get(key))
    meta.__setitem__ = Mock(side_effect=lambda key, value: data.update({key: value}))
    meta.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return meta


@pytest.fixture
def mock_meta_os():
    """Create a mock EyeEnfaceMeta for left eye (OS)."""
    meta = Mock()
    data = {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'scale_unit': 'mm',
        'laterality': 'OS'
    }
    meta.__getitem__ = Mock(side_effect=lambda key: data.get(key))
    meta.__setitem__ = Mock(side_effect=lambda key, value: data.update({key: value}))
    meta.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return meta


class TestLateralityPropertySetters:
    """Tests for laterality inference and validation when using property
    setters."""

    def test_set_fovea_after_optic_disc_infers_laterality(self, mock_meta_no_laterality):
        """Test that setting fovea after optic disc infers laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with only optic disc
        optic_disc_polygon = np.array([
            [40.0, 70.0],
            [40.0, 80.0],
            [50.0, 80.0],
            [50.0, 70.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=mock_meta_no_laterality, optic_disc=optic_disc, fovea=None)

        # Laterality should not be set yet
        assert enface.meta.get('laterality') is None

        # Now set fovea - optic disc is to the right of fovea -> should infer OD
        fovea_polygon = np.array([
            [45.0, 20.0],
            [45.0, 30.0],
            [55.0, 30.0],
            [55.0, 20.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface.fovea = fovea

        # Laterality should now be inferred as OD
        assert enface.laterality == 'OD'

    def test_set_optic_disc_after_fovea_infers_laterality(self, mock_meta_no_laterality):
        """Test that setting optic disc after fovea infers laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with only fovea
        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=mock_meta_no_laterality, optic_disc=None, fovea=fovea)

        # Laterality should not be set yet
        assert enface.meta.get('laterality') is None

        # Now set optic disc - optic disc is to the left of fovea -> should infer OS
        optic_disc_polygon = np.array([
            [40.0, 20.0],
            [40.0, 30.0],
            [50.0, 30.0],
            [50.0, 20.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        enface.optic_disc = optic_disc

        # Laterality should now be inferred as OS
        assert enface.laterality == 'OS'

    def test_set_fovea_validates_existing_laterality(self, mock_meta_od):
        """Test that setting fovea validates against existing laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with only optic disc and laterality=OD
        optic_disc_polygon = np.array([
            [40.0, 70.0],
            [40.0, 80.0],
            [50.0, 80.0],
            [50.0, 70.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=None)

        # Try to set fovea in wrong position (to the right of optic disc)
        fovea_polygon = np.array([
            [45.0, 90.0],
            [45.0, 100.0],
            [55.0, 100.0],
            [55.0, 90.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        # Should raise ValueError because OD requires optic disc to the right of fovea
        with pytest.raises(ValueError, match='Laterality mismatch.*Right eye'):
            enface.fovea = fovea

    def test_set_optic_disc_validates_existing_laterality(self, mock_meta_os):
        """Test that setting optic disc validates against existing laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with only fovea and laterality=OS
        fovea_polygon = np.array([
            [45.0, 70.0],
            [45.0, 80.0],
            [55.0, 80.0],
            [55.0, 70.0]
        ])
        fovea = EyeEnfaceFoveaAnnotation(fovea_polygon, shape=(100, 100))

        enface = EyeEnface(data=data, meta=mock_meta_os, optic_disc=None, fovea=fovea)

        # Try to set optic disc in wrong position (to the right of fovea)
        optic_disc_polygon = np.array([
            [40.0, 90.0],
            [40.0, 100.0],
            [50.0, 100.0],
            [50.0, 90.0]
        ])
        optic_disc = EyeEnfaceOpticDiscAnnotation(optic_disc_polygon, shape=(100, 100))

        # Should raise ValueError because OS requires optic disc to the left of fovea
        with pytest.raises(ValueError, match='Laterality mismatch.*Left eye'):
            enface.optic_disc = optic_disc

    def test_replace_fovea_revalidates(self, mock_meta_od):
        """Test that replacing fovea re-validates laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with correct anatomy for OD
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

        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)

        # Replace fovea with one in the wrong position
        wrong_fovea_polygon = np.array([
            [45.0, 90.0],
            [45.0, 100.0],
            [55.0, 100.0],
            [55.0, 90.0]
        ])
        wrong_fovea = EyeEnfaceFoveaAnnotation(wrong_fovea_polygon, shape=(100, 100))

        # Should raise ValueError
        with pytest.raises(ValueError, match='Laterality mismatch.*Right eye'):
            enface.fovea = wrong_fovea

    def test_replace_optic_disc_revalidates(self, mock_meta_os):
        """Test that replacing optic disc re-validates laterality."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with correct anatomy for OS
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

        enface = EyeEnface(data=data, meta=mock_meta_os, optic_disc=optic_disc, fovea=fovea)

        # Replace optic disc with one in the wrong position
        wrong_optic_disc_polygon = np.array([
            [40.0, 90.0],
            [40.0, 100.0],
            [50.0, 100.0],
            [50.0, 90.0]
        ])
        wrong_optic_disc = EyeEnfaceOpticDiscAnnotation(wrong_optic_disc_polygon, shape=(100, 100))

        # Should raise ValueError
        with pytest.raises(ValueError, match='Laterality mismatch.*Left eye'):
            enface.optic_disc = wrong_optic_disc

    def test_setting_none_fovea_does_not_validate(self, mock_meta_od):
        """Test that setting fovea to None doesn't trigger validation."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with both annotations
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

        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)

        # Should not raise
        enface.fovea = None
        assert enface.fovea is None

    def test_setting_none_optic_disc_does_not_validate(self, mock_meta_od):
        """Test that setting optic disc to None doesn't trigger validation."""
        data = np.arange(10000).reshape(100, 100).astype(np.int64)

        # Create enface with both annotations
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

        enface = EyeEnface(data=data, meta=mock_meta_od, optic_disc=optic_disc, fovea=fovea)

        # Should not raise
        enface.optic_disc = None
        assert enface.optic_disc is None
