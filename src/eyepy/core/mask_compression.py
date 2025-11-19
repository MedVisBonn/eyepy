"""Utilities for compressing and decompressing boolean masks using packbits.

This module provides functions to compress boolean masks using
numpy.packbits for efficient storage and to decompress them back to
their original form.
"""

import json
from typing import Optional

import numpy as np
import numpy.typing as npt


def compress_boolean_mask(mask: npt.NDArray[np.bool_]) -> tuple[bytes, dict]:
    """Compress a boolean mask using packbits.

    Args:
        mask: A boolean numpy array of any shape

    Returns:
        A tuple of (compressed_data, metadata) where:
        - compressed_data: bytes from np.packbits(mask.ravel())
        - metadata: dict containing 'shape' to reconstruct the original shape
    """
    # Flatten the mask
    flat_mask = mask.ravel()

    # Pack bits
    packed = np.packbits(flat_mask)

    # Store metadata
    metadata = {
        'shape': list(mask.shape),
        'dtype': 'bool',
    }

    return packed.tobytes(), metadata


def decompress_boolean_mask(
    compressed_data: bytes,
    metadata: dict,
) -> npt.NDArray[np.bool_]:
    """Decompress a boolean mask from packbits format.

    Args:
        compressed_data: bytes from np.packbits
        metadata: dict containing 'shape' from compress_boolean_mask

    Returns:
        The decompressed boolean mask with original shape
    """
    # Convert bytes back to packed array
    packed = np.frombuffer(compressed_data, dtype=np.uint8)

    # Unpack bits
    shape = metadata['shape']
    total_elements = np.prod(shape)
    unpacked = np.unpackbits(packed)[:total_elements]

    # Reshape to original shape
    mask = unpacked.astype(bool).reshape(shape)

    return mask


def is_boolean_array(arr: npt.NDArray) -> bool:
    """Check if an array is of boolean dtype.

    Args:
        arr: numpy array to check

    Returns:
        True if array dtype is bool, False otherwise
    """
    return arr.dtype == np.bool_
