"""Comprehensive backward compatibility tests for all annotation storage formats.

This module tests loading of:
1. Original stacked format (voxel_maps.npy / pixel_maps.npy)
2. Intermediate individual compressed format (voxel_map_*.bin with separate metadata)
3. New individual mixed-type format (voxel_map_*.npy or .bin per annotation)
"""

import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.mask_compression import compress_boolean_mask


class TestBackwardCompatibilityFormats:
    """Test loading annotations from all previous storage formats."""

    def _create_test_volume(self):
        """Create a basic test volume."""
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
        return ep.EyeVolume(data=data, meta=volume_meta)

    def test_format_1_stacked_npy_boolean_masks(self, tmp_path):
        """Test loading Format 1: Stacked NPY with boolean masks (original format).

        This was the original format where all boolean masks were stacked into a single npy file.
        """
        volume = self._create_test_volume()
        n_bscans, bscan_height, bscan_width = volume.shape

        # Create test masks
        mask1 = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.6
        mask2 = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.6

        # Manually create Format 1 file (stacked npy)
        save_path = tmp_path / 'format_1_stacked.eye'

        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_STORED) as zipf:
            # Save basic files
            volume_bytes = io.BytesIO()
            np.save(volume_bytes, volume._raw_data)
            zipf.writestr('raw_volume.npy', volume_bytes.getvalue())
            zipf.writestr('meta.json', json.dumps(volume.meta.as_dict()))

            # Save localizer
            localizer_bytes = io.BytesIO()
            np.save(localizer_bytes, volume.localizer.data)
            zipf.writestr('localizer/localizer.npy', localizer_bytes.getvalue())
            zipf.writestr('localizer/meta.json', json.dumps(volume.localizer.meta.as_dict()))

            # Format 1: Stacked voxel masks in single npy file
            stacked_voxels = np.stack([mask1, mask2])
            voxel_bytes = io.BytesIO()
            np.save(voxel_bytes, stacked_voxels)
            zipf.writestr('annotations/voxels/voxel_maps.npy', voxel_bytes.getvalue())
            zipf.writestr('annotations/voxels/meta.json',
                         json.dumps([{'name': 'mask1'}, {'name': 'mask2'}]))

            # Format 1: Stacked pixel masks in single npy file
            localizer_mask1 = np.random.rand(volume.localizer.size_y, volume.localizer.size_x) > 0.6
            localizer_mask2 = np.random.rand(volume.localizer.size_y, volume.localizer.size_x) > 0.6
            stacked_pixels = np.stack([localizer_mask1, localizer_mask2])
            pixel_bytes = io.BytesIO()
            np.save(pixel_bytes, stacked_pixels)
            zipf.writestr('localizer/annotations/pixel/pixel_maps.npy', pixel_bytes.getvalue())
            zipf.writestr('localizer/annotations/pixel/meta.json',
                         json.dumps([{'name': 'area1'}, {'name': 'area2'}]))

        # Load and verify
        loaded = ep.EyeVolume.load(save_path)

        assert 'mask1' in loaded.volume_maps
        assert 'mask2' in loaded.volume_maps
        np.testing.assert_array_equal(mask1, loaded.volume_maps['mask1'].data)
        np.testing.assert_array_equal(mask2, loaded.volume_maps['mask2'].data)

        assert len(loaded.localizer._area_maps) == 2
        names = [am.meta['name'] for am in loaded.localizer._area_maps]
        assert 'area1' in names and 'area2' in names

    def test_format_2_individual_compressed_boolean(self, tmp_path):
        """Test loading Format 2: Individual compressed boolean masks.

        This was an intermediate format where each boolean mask was individually
        compressed with packbits, but metadata was stored separately.
        """
        volume = self._create_test_volume()
        n_bscans, bscan_height, bscan_width = volume.shape

        # Create test masks
        mask1 = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.6
        mask2 = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.6

        # Manually create Format 2 file (individual compressed)
        save_path = tmp_path / 'format_2_compressed.eye'

        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_STORED) as zipf:
            # Save basic files
            volume_bytes = io.BytesIO()
            np.save(volume_bytes, volume._raw_data)
            zipf.writestr('raw_volume.npy', volume_bytes.getvalue())
            zipf.writestr('meta.json', json.dumps(volume.meta.as_dict()))

            # Save localizer
            localizer_bytes = io.BytesIO()
            np.save(localizer_bytes, volume.localizer.data)
            zipf.writestr('localizer/localizer.npy', localizer_bytes.getvalue())
            zipf.writestr('localizer/meta.json', json.dumps(volume.localizer.meta.as_dict()))

            # Format 2: Individual compressed masks with compression_meta in metadata
            for i, mask in enumerate([mask1, mask2]):
                compressed_data, compression_meta = compress_boolean_mask(mask)
                zipf.writestr(f'annotations/voxels/voxel_map_{i}.bin', compressed_data)
                zipf.writestr(f'annotations/voxels/voxel_map_{i}_meta.json',
                             json.dumps({
                                 'annotation_meta': {'name': f'mask{i+1}'},
                                 'compression_meta': compression_meta
                             }))

            # Similar for localizer
            localizer_mask1 = np.random.rand(volume.localizer.size_y, volume.localizer.size_x) > 0.6
            localizer_mask2 = np.random.rand(volume.localizer.size_y, volume.localizer.size_x) > 0.6
            for i, mask in enumerate([localizer_mask1, localizer_mask2]):
                compressed_data, compression_meta = compress_boolean_mask(mask)
                zipf.writestr(f'localizer/annotations/pixel/pixel_map_{i}.bin', compressed_data)
                zipf.writestr(f'localizer/annotations/pixel/pixel_map_{i}_meta.json',
                             json.dumps({
                                 'annotation_meta': {'name': f'area{i+1}'},
                                 'compression_meta': compression_meta
                             }))

        # Load and verify
        loaded = ep.EyeVolume.load(save_path)

        assert 'mask1' in loaded.volume_maps
        assert 'mask2' in loaded.volume_maps
        np.testing.assert_array_equal(mask1, loaded.volume_maps['mask1'].data)
        np.testing.assert_array_equal(mask2, loaded.volume_maps['mask2'].data)

        assert len(loaded.localizer._area_maps) == 2

    def test_format_3_individual_mixed_types(self, tmp_path):
        """Test loading Format 3: Individual mixed-type annotations (current format).

        This is the current format where each annotation is stored individually,
        with boolean masks compressed and other types as npy.
        """
        volume = self._create_test_volume()
        n_bscans, bscan_height, bscan_width = volume.shape

        # Create mixed type masks
        bool_mask = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.6
        float_mask = np.random.rand(n_bscans, bscan_height, bscan_width).astype(np.float32)
        int_mask = np.random.randint(0, 10, (n_bscans, bscan_height, bscan_width), dtype=np.int32)

        # Manually create Format 3 file (individual mixed types)
        save_path = tmp_path / 'format_3_mixed.eye'

        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_STORED) as zipf:
            # Save basic files
            volume_bytes = io.BytesIO()
            np.save(volume_bytes, volume._raw_data)
            zipf.writestr('raw_volume.npy', volume_bytes.getvalue())
            zipf.writestr('meta.json', json.dumps(volume.meta.as_dict()))

            # Save localizer
            localizer_bytes = io.BytesIO()
            np.save(localizer_bytes, volume.localizer.data)
            zipf.writestr('localizer/localizer.npy', localizer_bytes.getvalue())
            zipf.writestr('localizer/meta.json', json.dumps(volume.localizer.meta.as_dict()))

            # Format 3: Individual masks, each with its own storage
            # Boolean mask - compressed
            compressed_data, compression_meta = compress_boolean_mask(bool_mask)
            zipf.writestr('annotations/voxels/voxel_map_0.bin', compressed_data)
            zipf.writestr('annotations/voxels/voxel_map_0_meta.json',
                         json.dumps({
                             'annotation_meta': {'name': 'bool_mask'},
                             'compression_meta': compression_meta
                         }))

            # Float mask - npy
            float_bytes = io.BytesIO()
            np.save(float_bytes, float_mask)
            zipf.writestr('annotations/voxels/voxel_map_1.npy', float_bytes.getvalue())
            zipf.writestr('annotations/voxels/voxel_map_1_meta.json',
                         json.dumps({'name': 'float_mask'}))

            # Int mask - npy
            int_bytes = io.BytesIO()
            np.save(int_bytes, int_mask)
            zipf.writestr('annotations/voxels/voxel_map_2.npy', int_bytes.getvalue())
            zipf.writestr('annotations/voxels/voxel_map_2_meta.json',
                         json.dumps({'name': 'int_mask'}))

        # Load and verify
        loaded = ep.EyeVolume.load(save_path)

        assert 'bool_mask' in loaded.volume_maps
        assert 'float_mask' in loaded.volume_maps
        assert 'int_mask' in loaded.volume_maps

        np.testing.assert_array_equal(bool_mask, loaded.volume_maps['bool_mask'].data)
        assert loaded.volume_maps['bool_mask'].data.dtype == bool

        np.testing.assert_array_almost_equal(float_mask, loaded.volume_maps['float_mask'].data, decimal=6)
        assert loaded.volume_maps['float_mask'].data.dtype == np.float32

        np.testing.assert_array_equal(int_mask, loaded.volume_maps['int_mask'].data)
        assert loaded.volume_maps['int_mask'].data.dtype == np.int32
