"""Tests for saving and loading optic disc and fovea annotations with
EyeVolume."""
from pathlib import Path
import tempfile

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta


@pytest.fixture
def eyevolume_with_anatomical_annotations():
    """Create an EyeVolume with optic disc and fovea annotations."""
    # Create a simple volume
    volume_data = np.random.rand(10, 100, 200).astype(np.float32)

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, i * 0.1),
            end_pos=(1.0, i * 0.1),
            pos_unit='mm'
        ) for i in range(10)
    ]

    volume_meta = EyeVolumeMeta(
        scale_x=0.01,
        scale_y=0.01,
        scale_z=0.1,
        scale_unit='mm',
        bscan_meta=bscan_meta,
        laterality='OD'
    )

    # Create localizer with anatomical annotations
    localizer_data = np.random.rand(200, 200).astype(np.int64)
    localizer_meta = EyeEnfaceMeta(
        scale_x=0.01,
        scale_y=0.01,
        scale_unit='mm',
        laterality='OD'
    )

    # Create optic disc annotation (ellipse on the right side for OD)
    optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
        center=(100, 140),
        minor_axis=30,
        major_axis=35,
        rotation=0.2,
        shape=(200, 200)
    )

    # Create fovea annotation (small circle on the left side for OD)
    # Create a circular polygon for the fovea
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    fovea_radius = 5
    fovea_y = 100 + fovea_radius * np.sin(theta)
    fovea_x = 80 + fovea_radius * np.cos(theta)
    fovea_polygon = np.column_stack([fovea_y, fovea_x])
    fovea = EyeEnfaceFoveaAnnotation(
        polygon=fovea_polygon,
        shape=(200, 200)
    )

    localizer = EyeEnface(
        data=localizer_data,
        meta=localizer_meta,
        optic_disc=optic_disc,
        fovea=fovea
    )

    volume = ep.EyeVolume(
        data=volume_data,
        meta=volume_meta,
        localizer=localizer
    )

    return volume


class TestEyeVolumeSaveLoadAnatomicalAnnotations:
    """Test saving and loading of optic disc and fovea annotations."""

    def test_save_and_load_with_optic_disc_and_fovea(self, eyevolume_with_anatomical_annotations):
        """Test that optic disc and fovea are saved and loaded correctly."""
        volume = eyevolume_with_anatomical_annotations

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'

            # Save the volume
            volume.save(save_path)

            # Load the volume
            loaded_volume = ep.EyeVolume.load(save_path)

            # Verify optic disc is loaded
            assert loaded_volume.localizer.optic_disc is not None
            assert volume.localizer.optic_disc is not None

            # Verify fovea is loaded
            assert loaded_volume.localizer.fovea is not None
            assert volume.localizer.fovea is not None

            # Verify optic disc polygon matches
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.optic_disc.polygon,
                volume.localizer.optic_disc.polygon
            )

            # Verify fovea polygon matches
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.fovea.polygon,
                volume.localizer.fovea.polygon
            )

            # Verify optic disc center matches
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.optic_disc.center,
                volume.localizer.optic_disc.center
            )

            # Verify fovea center matches
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.fovea.center,
                volume.localizer.fovea.center
            )

    def test_save_and_load_with_optic_disc_only(self):
        """Test saving and loading when only optic disc is present."""
        # Create volume with only optic disc
        volume_data = np.random.rand(5, 50, 100).astype(np.float32)

        bscan_meta = [
            EyeBscanMeta(start_pos=(0, i * 0.1), end_pos=(1.0, i * 0.1), pos_unit='mm')
            for i in range(5)
        ]

        volume_meta = EyeVolumeMeta(
            scale_x=0.01, scale_y=0.01, scale_z=0.1, scale_unit='mm',
            bscan_meta=bscan_meta
        )

        localizer_data = np.random.rand(100, 100).astype(np.int64)
        localizer_meta = EyeEnfaceMeta(scale_x=0.01, scale_y=0.01, scale_unit='mm')

        optic_disc = EyeEnfaceOpticDiscAnnotation.from_ellipse(
            center=(50, 60), minor_axis=20, major_axis=25, shape=(100, 100)
        )

        localizer = EyeEnface(
            data=localizer_data, meta=localizer_meta, optic_disc=optic_disc
        )

        volume = ep.EyeVolume(data=volume_data, meta=volume_meta, localizer=localizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            assert loaded_volume.localizer.optic_disc is not None
            assert loaded_volume.localizer.fovea is None
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.optic_disc.polygon,
                volume.localizer.optic_disc.polygon
            )

    def test_save_and_load_with_fovea_only(self):
        """Test saving and loading when only fovea is present."""
        # Create volume with only fovea
        volume_data = np.random.rand(5, 50, 100).astype(np.float32)

        bscan_meta = [
            EyeBscanMeta(start_pos=(0, i * 0.1), end_pos=(1.0, i * 0.1), pos_unit='mm')
            for i in range(5)
        ]

        volume_meta = EyeVolumeMeta(
            scale_x=0.01, scale_y=0.01, scale_z=0.1, scale_unit='mm',
            bscan_meta=bscan_meta
        )

        localizer_data = np.random.rand(100, 100).astype(np.int64)
        localizer_meta = EyeEnfaceMeta(scale_x=0.01, scale_y=0.01, scale_unit='mm')

        # Create a circular polygon for the fovea
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        fovea_radius = 4
        fovea_y = 50 + fovea_radius * np.sin(theta)
        fovea_x = 40 + fovea_radius * np.cos(theta)
        fovea_polygon = np.column_stack([fovea_y, fovea_x])
        fovea = EyeEnfaceFoveaAnnotation(
            polygon=fovea_polygon,
            shape=(100, 100)
        )

        localizer = EyeEnface(
            data=localizer_data, meta=localizer_meta, fovea=fovea
        )

        volume = ep.EyeVolume(data=volume_data, meta=volume_meta, localizer=localizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            assert loaded_volume.localizer.optic_disc is None
            assert loaded_volume.localizer.fovea is not None
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer.fovea.polygon,
                volume.localizer.fovea.polygon
            )

    def test_save_and_load_without_anatomical_annotations(self):
        """Test saving and loading when no anatomical annotations are present."""
        # Create volume without optic disc and fovea
        volume_data = np.random.rand(5, 50, 100).astype(np.float32)

        bscan_meta = [
            EyeBscanMeta(start_pos=(0, i * 0.1), end_pos=(1.0, i * 0.1), pos_unit='mm')
            for i in range(5)
        ]

        volume_meta = EyeVolumeMeta(
            scale_x=0.01, scale_y=0.01, scale_z=0.1, scale_unit='mm',
            bscan_meta=bscan_meta
        )

        localizer_data = np.random.rand(100, 100).astype(np.int64)
        localizer_meta = EyeEnfaceMeta(scale_x=0.01, scale_y=0.01, scale_unit='mm')

        localizer = EyeEnface(data=localizer_data, meta=localizer_meta)

        volume = ep.EyeVolume(data=volume_data, meta=volume_meta, localizer=localizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            assert loaded_volume.localizer.optic_disc is None
            assert loaded_volume.localizer.fovea is None

    def test_save_preserves_shape_attribute(self, eyevolume_with_anatomical_annotations):
        """Test that the shape attribute is preserved during save/load."""
        volume = eyevolume_with_anatomical_annotations

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            # Verify shape is preserved
            assert loaded_volume.localizer.optic_disc.shape == volume.localizer.optic_disc.shape
            assert loaded_volume.localizer.fovea.shape == volume.localizer.fovea.shape

            # Verify we can generate masks (which requires shape)
            original_od_mask = volume.localizer.optic_disc.mask
            loaded_od_mask = loaded_volume.localizer.optic_disc.mask
            np.testing.assert_array_equal(original_od_mask, loaded_od_mask)

            original_fovea_mask = volume.localizer.fovea.mask
            loaded_fovea_mask = loaded_volume.localizer.fovea.mask
            np.testing.assert_array_equal(original_fovea_mask, loaded_fovea_mask)
