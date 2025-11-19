"""Test for mixed data type annotations - individual storage."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeVolumeMeta


def test_mixed_datatype_annotations():
    """Test that annotations with mixed data types are saved and loaded
    individually."""
    # Create a test volume
    n_bscans = 5
    bscan_height = 30
    bscan_width = 50
    data = np.random.random((n_bscans, bscan_height, bscan_width)).astype(np.float32)

    volume_meta = EyeVolumeMeta(
        scale_x=1.0,
        scale_y=1.0,
        scale_z=1.0,
        scale_unit='pixel',
        bscan_meta=[
            EyeBscanMeta(start_pos=(0, i * 0.5),
                        end_pos=(float(bscan_width - 1), i * 0.5),
                        pos_unit='pixel')
            for i in range(n_bscans)
        ],
        laterality='OD'
    )

    volume = ep.EyeVolume(data=data, meta=volume_meta)

    # Add mixed data type annotations
    print('Creating mixed data type annotations...')

    # Boolean mask (will be compressed)
    bool_mask = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.7
    volume.add_pixel_annotation(bool_mask, name='drusen_boolean')

    # Float mask (will be saved as npy)
    float_mask = np.random.rand(n_bscans, bscan_height, bscan_width).astype(np.float32)
    volume.add_pixel_annotation(float_mask, name='thickness_map')

    # Integer mask (will be saved as npy)
    int_mask = np.random.randint(0, 10, (n_bscans, bscan_height, bscan_width), dtype=np.int32)
    volume.add_pixel_annotation(int_mask, name='layer_index')

    print(f'  Boolean mask dtype: {bool_mask.dtype}')
    print(f'  Float mask dtype: {float_mask.dtype}')
    print(f'  Int mask dtype: {int_mask.dtype}')

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'mixed_types.eye'

        print(f'\nSaving volume...')
        volume.save(save_path, compress=False)

        print(f'Loading volume...')
        loaded_volume = ep.EyeVolume.load(save_path)

        # Verify all annotations are present and intact
        print('Verifying annotations...')

        # Check boolean mask
        assert 'drusen_boolean' in loaded_volume.volume_maps
        loaded_bool = loaded_volume.volume_maps['drusen_boolean'].data
        assert loaded_bool.dtype == bool
        np.testing.assert_array_equal(bool_mask, loaded_bool)
        print('  ✓ Boolean mask preserved with correct dtype')

        # Check float mask
        assert 'thickness_map' in loaded_volume.volume_maps
        loaded_float = loaded_volume.volume_maps['thickness_map'].data
        assert loaded_float.dtype == np.float32
        np.testing.assert_array_almost_equal(float_mask, loaded_float, decimal=6)
        print('  ✓ Float mask preserved with correct dtype')

        # Check integer mask
        assert 'layer_index' in loaded_volume.volume_maps
        loaded_int = loaded_volume.volume_maps['layer_index'].data
        assert loaded_int.dtype == np.int32
        np.testing.assert_array_equal(int_mask, loaded_int)
        print('  ✓ Integer mask preserved with correct dtype')

    print('\n✓ Mixed data type test passed!')


def test_mixed_datatype_localizer_annotations():
    """Test that localizer annotations with mixed data types are saved
    individually."""
    n_bscans = 5
    bscan_height = 30
    bscan_width = 50
    data = np.random.random((n_bscans, bscan_height, bscan_width)).astype(np.float32)

    volume_meta = EyeVolumeMeta(
        scale_x=1.0, scale_y=1.0, scale_z=1.0, scale_unit='pixel',
        bscan_meta=[
            EyeBscanMeta(start_pos=(0, i * 0.5),
                        end_pos=(float(bscan_width - 1), i * 0.5),
                        pos_unit='pixel')
            for i in range(n_bscans)
        ], laterality='OD'
    )
    volume = ep.EyeVolume(data=data, meta=volume_meta)

    # Add mixed data type annotations to localizer
    bool_area = np.random.rand(volume.localizer.size_y, volume.localizer.size_x) > 0.7
    volume.localizer.add_area_annotation(bool_area, name='optic_disc_bool')

    float_area = np.random.rand(volume.localizer.size_y, volume.localizer.size_x).astype(np.float32)
    volume.localizer.add_area_annotation(float_area, name='probability_map')

    print('Testing mixed type localizer annotations...')

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'mixed_localizer.eye'

        volume.save(save_path, compress=False)
        loaded_volume = ep.EyeVolume.load(save_path)

        # Verify both annotations
        found_bool = False
        found_float = False

        for area_map in loaded_volume.localizer._area_maps:
            if area_map.meta['name'] == 'optic_disc_bool':
                assert area_map.data.dtype == bool
                np.testing.assert_array_equal(bool_area, area_map.data)
                found_bool = True
            elif area_map.meta['name'] == 'probability_map':
                assert area_map.data.dtype == np.float32
                np.testing.assert_array_almost_equal(float_area, area_map.data, decimal=6)
                found_float = True

        assert found_bool, 'Boolean localizer annotation not found'
        assert found_float, 'Float localizer annotation not found'

    print('✓ Mixed type localizer annotations test passed!')


if __name__ == '__main__':
    test_mixed_datatype_annotations()
    test_mixed_datatype_localizer_annotations()
