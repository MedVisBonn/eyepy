
from unittest.mock import MagicMock

import numpy as np
import pytest

from eyepy.core.annotations import EyeVolumeSlabAnnotation


def test_slab_annotation_cache_invalidation():
    # Mock EyeVolume and Layers
    mock_volume = MagicMock()
    mock_volume.shape = (10, 100, 100) # (B-scans, Height, Width)
    mock_volume.size_y = 100

    # Create fake layer data
    layer1_data = np.full((10, 100), 20.0) # Flat layer at y=20
    layer2_data = np.full((10, 100), 40.0) # Flat layer at y=40
    layer3_data = np.full((10, 100), 60.0) # Flat layer at y=60

    mock_volume.layers = {
        'Layer1': MagicMock(data=layer1_data),
        'Layer2': MagicMock(data=layer2_data),
        'Layer3': MagicMock(data=layer3_data)
    }

    # Initialize Slab Annotation
    slab = EyeVolumeSlabAnnotation(mock_volume, top_layer='Layer1', bottom_layer='Layer2')

    # Access mask to trigger computation and caching
    mask1 = slab.mask
    # Mask should be True between 20 and 40.
    # Check a pixel at y=30.
    assert mask1[0, 30, 0] == True, 'Initial mask should enable pixel at y=30'

    # Change bottom layer to Layer3 (y=60)
    slab.bottom_layer = 'Layer3'

    # Access mask again
    mask2 = slab.mask

    # Check a pixel at y=50.
    # It should be masked now (between 20 and 60).
    # If cache is not invalidated, it will assume old mask (20 to 40) and y=50 will be False.
    assert mask2[0, 50, 0] == True, 'Updated mask should enable pixel at y=50 after changing bottom_layer'
