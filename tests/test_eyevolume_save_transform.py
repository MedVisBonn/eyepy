"""Tests for saving and loading localizer_transform with EyeVolume."""
from pathlib import Path
import tempfile

import numpy as np
import pytest
from skimage import transform

import eyepy as ep
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta


@pytest.fixture
def eyevolume_with_custom_transform():
    """Create an EyeVolume with a custom localizer transform."""
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
    )

    # Create localizer
    localizer_data = np.random.rand(200, 200).astype(np.int64)
    localizer_meta = EyeEnfaceMeta(
        scale_x=0.01,
        scale_y=0.01,
        scale_unit='mm',
    )

    localizer = EyeEnface(data=localizer_data, meta=localizer_meta)

    # Create a custom affine transform (different from the default)
    # This simulates a user manually setting a custom transformation
    custom_transform = transform.AffineTransform(
        scale=(1.5, 1.2),
        rotation=0.1,
        translation=(10, 5)
    )

    volume = ep.EyeVolume(
        data=volume_data,
        meta=volume_meta,
        localizer=localizer,
        transformation=custom_transform
    )

    return volume


class TestEyeVolumeSaveLoadTransform:
    """Test saving and loading of localizer_transform."""

    def test_save_and_load_custom_transform(self, eyevolume_with_custom_transform):
        """Test that a custom localizer_transform is saved and loaded
        correctly."""
        volume = eyevolume_with_custom_transform
        original_params = volume.localizer_transform.params.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'

            # Save the volume
            volume.save(save_path)

            # Load the volume
            loaded_volume = ep.EyeVolume.load(save_path)

            # Verify the transform parameters match
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer_transform.params,
                original_params
            )

            # Verify it's not the default transform
            # (the default would be different from our custom one)
            default_volume = ep.EyeVolume(
                data=volume._raw_data,
                meta=volume.meta,
                localizer=volume.localizer
            )

            # The loaded transform should match the original, not the default
            assert not np.allclose(
                loaded_volume.localizer_transform.params,
                default_volume.localizer_transform.params
            ), 'Loaded transform should be custom, not default'

    def test_save_and_load_default_transform(self):
        """Test that the default computed transform is saved and loaded
        correctly."""
        # Create a volume with default transform (no custom transformation provided)
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

        # Create volume without custom transformation
        volume = ep.EyeVolume(data=volume_data, meta=volume_meta, localizer=localizer)
        original_params = volume.localizer_transform.params.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            # The default transform should also be preserved
            np.testing.assert_array_almost_equal(
                loaded_volume.localizer_transform.params,
                original_params
            )

    def test_backward_compatibility_no_transform_file(self):
        """Test that loading old files without transform_params.npy still
        works."""
        # Create and save a volume
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

            # Manually remove the transform file to simulate an old save format
            import shutil
            import zipfile

            # Extract the archive
            extract_path = Path(tmpdir) / 'extracted'
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Remove the transform file
            transform_file = extract_path / 'localizer' / 'transform_params.npy'
            if transform_file.exists():
                transform_file.unlink()

            # Re-create the archive
            old_format_path = Path(tmpdir) / 'old_format_volume.eye'
            shutil.make_archive(str(old_format_path), 'zip', root_dir=str(extract_path))
            shutil.move(str(old_format_path) + '.zip', old_format_path)

            # Load should still work and compute the transform
            loaded_volume = ep.EyeVolume.load(old_format_path)

            # Should have a valid transform (computed, not loaded)
            assert loaded_volume.localizer_transform is not None
            assert loaded_volume.localizer_transform.params.shape == (3, 3)

    def test_transform_inverse_works_after_load(self, eyevolume_with_custom_transform):
        """Test that the inverse transform works correctly after loading."""
        volume = eyevolume_with_custom_transform

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_volume.eye'
            volume.save(save_path)
            loaded_volume = ep.EyeVolume.load(save_path)

            # Test that we can use the transform
            test_point = np.array([[50, 100]])  # A point in volume space

            # Transform using original
            transformed_orig = volume.localizer_transform(test_point)

            # Transform using loaded
            transformed_loaded = loaded_volume.localizer_transform(test_point)

            # Should be the same
            np.testing.assert_array_almost_equal(
                transformed_orig,
                transformed_loaded
            )

            # Test inverse also works
            inverse_orig = volume.localizer_transform.inverse(transformed_orig)
            inverse_loaded = loaded_volume.localizer_transform.inverse(transformed_loaded)

            np.testing.assert_array_almost_equal(inverse_orig, inverse_loaded)
            np.testing.assert_array_almost_equal(inverse_orig, test_point)

    def test_different_transform_types(self):
        """Test saving/loading with different types of affine transforms."""
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

        # Test different transform configurations
        test_transforms = [
            transform.AffineTransform(scale=(2, 2)),
            transform.AffineTransform(rotation=np.pi/4),
            transform.AffineTransform(translation=(50, 30)),
            transform.AffineTransform(scale=(1.5, 2), rotation=0.3, translation=(10, 20)),
        ]

        for custom_transform in test_transforms:
            volume = ep.EyeVolume(
                data=volume_data,
                meta=volume_meta,
                localizer=localizer,
                transformation=custom_transform
            )
            original_params = volume.localizer_transform.params.copy()

            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / 'test_volume.eye'
                volume.save(save_path)
                loaded_volume = ep.EyeVolume.load(save_path)

                np.testing.assert_array_almost_equal(
                    loaded_volume.localizer_transform.params,
                    original_params,
                    err_msg=f"Failed for transform with params:\n{original_params}"
                )
