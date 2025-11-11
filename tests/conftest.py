import numpy as np
import pytest

import eyepy as ep
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta


@pytest.fixture(scope='module')
def eyevolume():
    """Create a test EyeVolume with comprehensive metadata and annotations.

    This fixture includes:
    - Volume data and metadata (scale, laterality, bscan_meta)
    - Layer annotations
    - Volume map (voxel) annotations
    - Slab annotations
    - Area (pixel) annotations on the localizer
    - Localizer image with metadata
    """
    n_bscans = 10
    bscan_height = 50
    bscan_width = 100
    data = np.random.random(
        (n_bscans, bscan_height, bscan_width)).astype(np.float32)

    # Create detailed metadata with default scale values (pixel units)
    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, i * 0.5),
            end_pos=(float(bscan_width - 1), i * 0.5),
            pos_unit='pixel'
        ) for i in range(n_bscans)
    ]

    volume_meta = EyeVolumeMeta(
        scale_x=1.0,  # Default pixel scale
        scale_y=1.0,  # Default pixel scale
        scale_z=1.0,  # Default pixel scale
        scale_unit='pixel',
        bscan_meta=bscan_meta,
        laterality='OD'
    )

    # Create volume
    volume = ep.EyeVolume(data=data, meta=volume_meta)

    # Add layer annotations
    volume.add_layer_annotation(
        np.random.rand(n_bscans, bscan_width) * bscan_height,
        name='ILM',
        current_color='#FF0000'
    )
    volume.add_layer_annotation(
        np.random.rand(n_bscans, bscan_width) * bscan_height,
        name='RPE',
        current_color='#00FF00'
    )

    # Add volume map (voxel) annotation
    voxel_map = np.random.rand(n_bscans, bscan_height, bscan_width) > 0.8
    volume.add_pixel_annotation(voxel_map, name='drusen')

    # Add slab annotation
    volume.add_slab_annotation(name='retina', top_layer='ILM', bottom_layer='RPE')

    # Create localizer with metadata
    localizer_data = np.random.randint(0, 256, (bscan_width, bscan_width), dtype=np.int64)
    localizer_meta = EyeEnfaceMeta(
        scale_x=1.0,  # Default pixel scale
        scale_y=1.0,  # Default pixel scale
        scale_unit='pixel',
        laterality='OD'
    )
    localizer = EyeEnface(data=localizer_data, meta=localizer_meta)

    # Add area annotation to localizer
    area_map = np.random.rand(bscan_width, bscan_width) > 0.7
    localizer.add_area_annotation(area_map, name='optic_disc_region')

    # Update volume's localizer
    volume.localizer = localizer

    return volume
