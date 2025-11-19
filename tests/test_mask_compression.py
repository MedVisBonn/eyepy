"""Tests for boolean mask compression functionality."""

import numpy as np
import pytest

from eyepy.core.mask_compression import compress_boolean_mask
from eyepy.core.mask_compression import decompress_boolean_mask
from eyepy.core.mask_compression import is_boolean_array


class TestMaskCompressionUtilities:
    """Test basic mask compression utility functions."""

    def test_is_boolean_array(self):
        """Test boolean array detection."""
        bool_array = np.array([True, False, True], dtype=bool)
        assert is_boolean_array(bool_array) is True

        int_array = np.array([1, 0, 1], dtype=int)
        assert is_boolean_array(int_array) is False

        float_array = np.array([1.0, 0.0, 1.0], dtype=float)
        assert is_boolean_array(float_array) is False

    def test_compress_decompress_1d(self):
        """Test compression and decompression of 1D boolean array."""
        original = np.array([True, False, True, False, True], dtype=bool)
        compressed_data, metadata = compress_boolean_mask(original)
        decompressed = decompress_boolean_mask(compressed_data, metadata)

        np.testing.assert_array_equal(original, decompressed)
        assert metadata['shape'] == list(original.shape)

    def test_compress_decompress_2d(self):
        """Test compression and decompression of 2D boolean array."""
        original = np.random.rand(50, 60) > 0.5
        compressed_data, metadata = compress_boolean_mask(original)
        decompressed = decompress_boolean_mask(compressed_data, metadata)

        np.testing.assert_array_equal(original, decompressed)
        assert metadata['shape'] == list(original.shape)

    def test_compress_decompress_3d(self):
        """Test compression and decompression of 3D boolean array."""
        original = np.random.rand(10, 50, 100) > 0.5
        compressed_data, metadata = compress_boolean_mask(original)
        decompressed = decompress_boolean_mask(compressed_data, metadata)

        np.testing.assert_array_equal(original, decompressed)
        assert metadata['shape'] == list(original.shape)

    def test_compress_all_true(self):
        """Test compression of array of all True values."""
        original = np.ones((100, 100), dtype=bool)
        compressed_data, metadata = compress_boolean_mask(original)
        decompressed = decompress_boolean_mask(compressed_data, metadata)

        np.testing.assert_array_equal(original, decompressed)

    def test_compress_all_false(self):
        """Test compression of array of all False values."""
        original = np.zeros((100, 100), dtype=bool)
        compressed_data, metadata = compress_boolean_mask(original)
        decompressed = decompress_boolean_mask(compressed_data, metadata)

        np.testing.assert_array_equal(original, decompressed)

    def test_compression_ratio(self):
        """Verify that compression actually reduces size."""
        # Create a large boolean array
        original = np.random.rand(100, 100, 100) > 0.5

        # Compute uncompressed size (approximate)
        uncompressed_size = original.nbytes  # bool is 1 byte per element

        # Compress and check size
        compressed_data, metadata = compress_boolean_mask(original)
        compressed_size = len(compressed_data)

        # packbits should compress to ~1/8 of original size
        expected_compression_ratio = uncompressed_size / 8
        assert compressed_size < expected_compression_ratio * 1.1, \
            f'Compression ratio not as expected: {uncompressed_size} -> {compressed_size} bytes'

    def test_metadata_preserved(self):
        """Test that metadata is correctly preserved."""
        original = np.random.rand(10, 20, 30) > 0.5
        compressed_data, metadata = compress_boolean_mask(original)

        assert 'shape' in metadata
        assert 'dtype' in metadata
        assert metadata['dtype'] == 'bool'
        assert metadata['shape'] == [10, 20, 30]

    def test_unpackbits_precision(self):
        """Test that unpack precision is maintained with non-byte-aligned
        sizes."""
        # Test various sizes that might not be byte-aligned
        for size in [7, 9, 15, 17, 63, 64, 65, 100, 127, 128, 129]:
            original = np.random.rand(size) > 0.5
            compressed_data, metadata = compress_boolean_mask(original)
            decompressed = decompress_boolean_mask(compressed_data, metadata)

            np.testing.assert_array_equal(original, decompressed,
                                          err_msg=f'Failed for size {size}')
