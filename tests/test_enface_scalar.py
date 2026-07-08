import numpy as np

import eyepy as ep
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.quant.enface_scalar import EnfaceScalarQuantification
from eyepy.quant.thickness import thickness_from_layer_pair
from eyepy.quant.thickness import thickness_from_voxel_annotation


def _make_thickness_volume():
    n_bscans = 4
    bscan_height = 20
    bscan_width = 8
    data = np.zeros((n_bscans, bscan_height, bscan_width), dtype=np.float32)

    bscan_meta = [
        EyeBscanMeta(start_pos=(0, i), end_pos=(bscan_width - 1, i), pos_unit='pixel')
        for i in range(n_bscans)
    ]
    meta = EyeVolumeMeta(
        scale_x=1.0,
        scale_y=2.0,
        scale_z=1.0,
        scale_unit='mm',
        bscan_meta=bscan_meta,
        laterality='OD',
    )
    volume = ep.EyeVolume(data=data, meta=meta)

    volume.add_layer_annotation(
        np.full((n_bscans, bscan_width), 5.0),
        name='ILM',
    )
    volume.add_layer_annotation(
        np.full((n_bscans, bscan_width), 15.0),
        name='BM',
    )

    mask = np.zeros((n_bscans, bscan_height, bscan_width), dtype=bool)
    mask[:, 4:8, 2] = True
    volume.add_pixel_annotation(mask, name='drusen')

    localizer = ep.EyeEnface(
        data=np.zeros((bscan_width, bscan_width), dtype=np.int64),
        meta=EyeEnfaceMeta(scale_x=1.0, scale_y=1.0, scale_unit='mm', laterality='OD'),
    )
    volume.localizer = localizer
    return volume


def test_from_layer_pair_unit_and_map():
    volume = _make_thickness_volume()
    thickness = EnfaceScalarQuantification.from_layer_pair(volume, 'ILM', 'BM')

    assert thickness.unit == 'mm'
    assert thickness.data.shape == volume.localizer.shape

    bscan_map = thickness_from_layer_pair(
        volume.layers['ILM'].data,
        volume.layers['BM'].data,
        volume.scale_y,
    )
    np.testing.assert_allclose(bscan_map, 20.0)


def test_quantify_thickness_convenience():
    volume = _make_thickness_volume()
    thickness = volume.quantify_thickness('ILM', 'BM')

    assert thickness.unit == 'mm'
    assert 'Laterality' in thickness.quantification
    assert thickness.quantification['Laterality'] == 'OD'


def test_quantify_thickness_with_etdrs_preset():
    from eyepy.quant.grid_presets import ETDRS_9

    volume = _make_thickness_volume()
    thickness = volume.quantify_thickness('ILM', 'BM', grid=ETDRS_9)

    assert 'Central [mm]' in thickness.quantification
    assert 'Inner Superior [mm]' in thickness.quantification
    assert 'Outer Temporal [mm]' in thickness.quantification


def test_from_pixel_annotation_matches_projection_scaling():
    volume = _make_thickness_volume()
    thickness = EnfaceScalarQuantification.from_pixel_annotation(volume, 'drusen')

    assert thickness.unit == 'mm'
    annotation = volume.volume_maps['drusen']
    expected_column = 4 * volume.scale_y
    bscan_map = thickness_from_voxel_annotation(
        annotation.data,
        volume.scale_y,
    )
    np.testing.assert_allclose(bscan_map[:, 2], expected_column)
    assert bscan_map[:, [0, 1, 3, 4, 5, 6, 7]].sum() == 0

    # Physical-unit version of the depth projection (projection is flip(sum) in voxels)
    np.testing.assert_allclose(
        annotation.projection * volume.scale_y,
        np.flip(bscan_map, axis=0),
    )


def test_pixel_annotation_thickness_quantification_property():
    volume = _make_thickness_volume()
    annotation = volume.volume_maps['drusen']

    result = annotation.thickness_quantification

    assert 'Laterality' in result
    assert any(key.endswith('[mm]') for key in result if key != 'Laterality')


def test_plot_thickness_uses_annotation_grid_preset():
    from eyepy.quant.grid_presets import ETDRS_9

    volume = _make_thickness_volume()
    annotation = volume.volume_maps['drusen']
    annotation.grid_preset = ETDRS_9

    thickness = annotation.enface_scalar_quantification()

    assert thickness.grid_preset is ETDRS_9
    assert 'Central' in thickness.masks
    assert 'Outer Temporal' in thickness.masks
