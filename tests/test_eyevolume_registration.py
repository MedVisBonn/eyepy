import struct
from types import SimpleNamespace

import numpy as np
import pytest
from skimage.transform import AffineTransform

from eyepy import EyeBscanMeta
from eyepy import EyeEnface
from eyepy import EyeEnfaceMeta
from eyepy import EyeVolume
from eyepy import EyeVolumeMeta
from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.io.he.e2e_reader import E2ESeriesStructure


def matrix(tx=0, ty=0, shear=0):
    return [[1, 0, tx], [shear, 1, ty]]


def registration_payload(values):
    raw = bytearray(100)
    for offset, value in zip([4, 8, 12, 28, 32, 36],
                             np.asarray(values).ravel()):
        struct.pack_into('<f', raw, offset, value)
    return bytes(raw)


def folder(payload):
    return SimpleNamespace(get_bytes=lambda: payload)


def series_fixture():
    series = E2ESeriesStructure(1)
    table = bytes(12) + struct.pack('<I2f', 2, 2, 4) + b'trailer'
    series.folders[10010] = [folder(table)]
    for slice_id, akt_image, numerator in [(0, 1, 8), (2, 0, 6)]:
        meta = SimpleNamespace(aktImage=akt_image, unknown1=numerator)
        series.substructure[slice_id] = SimpleNamespace(
            _get_bscanmeta_data=lambda m=meta: m,
            folders={10012: [folder(registration_payload(matrix()))]})
    return series


@pytest.fixture
def transformed_volume():
    source = np.full((2, 4, 6), .25, np.float32)
    source[0, 0, :2] = [.75, 0]
    source[1, 0, 0] = 1.9990234375
    meta = EyeVolumeMeta(scale_x=1,
                         scale_y=1,
                         scale_z=1,
                         scale_unit='pixel',
                         intensity_transform='vol',
                         bscan_meta=[
                             EyeBscanMeta(start_pos=(0, i),
                                          end_pos=(5, i),
                                          pos_unit='pixel',
                                          scan_pattern=3) for i in range(2)
                         ])
    localizer = EyeEnface(
        np.zeros((6, 6), np.uint8),
        EyeEnfaceMeta(scale_x=1, scale_y=1, scale_unit='pixel'))
    return EyeVolume(source,
                     meta,
                     localizer,
                     AffineTransform(),
                     spatial_transform=HeidelbergBscanTransform(
                         [2, 3], [matrix(tx=.5), matrix()]))


def test_gain_indexes_table_by_akt_image_instead_of_slice_order():
    transform = series_fixture().get_spatial_transform()
    np.testing.assert_array_equal(transform.gains, [2, 3])


@pytest.mark.parametrize('failure',
                         ['table', 'matrix', 'index', 'denominator', 'affine'])
def test_unavailable_registration_warns_instead_of_applying_partial_transform(
        failure):
    series = series_fixture()
    if failure == 'table':
        series.folders.pop(10010)
    elif failure == 'matrix':
        series.slices[0].folders[10012] = [folder(bytes(20))]
    elif failure == 'index':
        series.slices[0]._get_bscanmeta_data = lambda: SimpleNamespace(
            aktImage=3, unknown1=8)
    elif failure == 'denominator':
        series.folders[10010] = [
            folder(bytes(12) + struct.pack('<I2f', 2, 0, 4))
        ]
    else:
        values = matrix()
        values[0][0] = 2
        series.slices[0].folders[10012] = [folder(registration_payload(values))]
    with pytest.warns(RuntimeWarning, match='Keeping the unregistered'):
        assert series.get_spatial_transform() is None


def test_normalization_precedes_interpolation_and_display(transformed_volume):
    volume = transformed_volume
    original = volume.raw_data
    # clip(.75 * 2) = 1 BEFORE interpolating halfway towards 0: result .5.
    assert volume[0].linear_data[0, 0] == .5
    assert volume[0].data[0, 0] == 234
    assert volume[1].linear_data[0, 0] == 0  # Raw 0xFFFF renders black.
    assert volume.raw_data[1, 0, 0] == 1.9990234375
    np.testing.assert_array_equal(volume.linear_data[0], volume[0].linear_data)
    np.testing.assert_array_equal(volume.data[0], volume[0].data)
    np.testing.assert_array_equal(volume.raw_data, original)


def test_single_bscan_does_not_register_the_whole_volume(
        transformed_volume, monkeypatch):
    transform = transformed_volume.spatial_transform
    calls = []
    register = transform.register

    def record(data, index):
        calls.append(index)
        return register(data, index)

    monkeypatch.setattr(transform, 'register', record)
    transformed_volume[1].data
    assert calls == [1]
    assert transformed_volume._data is None


def test_registered_layer_coordinates_and_original_snapshot_survive_save(
        transformed_volume, tmp_path):
    volume = transformed_volume
    raw_heights = np.full((2, 6), 2.0)
    heights = volume.spatial_transform.transform_layers(raw_heights)
    layer = volume.add_layer_annotation(heights,
                                        name='ILM',
                                        raw_data=raw_heights)
    assert np.isnan(layer.data[0, -1])
    layer.data[0, 0] = 1.5
    np.testing.assert_array_equal(layer.raw_data, raw_heights)
    raw_copy = layer.raw_data
    raw_copy[:] = 0
    assert layer.raw_data[0, 0] == 2
    path = tmp_path / 'registered.eye'
    volume.save(path)
    loaded = EyeVolume.load(path)
    np.testing.assert_array_equal(loaded.raw_data, volume.raw_data)
    np.testing.assert_array_equal(loaded.linear_data.view('<u4'),
                                  volume.linear_data.view('<u4'))
    np.testing.assert_array_equal(loaded.data, volume.data)
    np.testing.assert_array_equal(loaded.layers['ILM'].data, layer.data)
    np.testing.assert_array_equal(loaded.layers['ILM'].raw_data, raw_heights)
    assert loaded.meta == volume.meta


def test_e2e_reader_attaches_transform_and_aligns_layers(
        transformed_volume, monkeypatch):
    series = series_fixture()
    volume = transformed_volume
    raw = volume.raw_data
    raw_heights = np.full((2, 6), 2.)
    monkeypatch.setattr(series, 'get_meta', lambda: volume.meta.copy())
    monkeypatch.setattr(series, 'get_bscans', lambda: raw)
    monkeypatch.setattr(series, 'get_localizer', lambda registered=True: volume.localizer)
    monkeypatch.setattr(series, 'get_layers', lambda: {0: raw_heights})
    monkeypatch.setattr(series, 'get_layer_manual_maps',
                        lambda: {0: [True, False]})
    imported = series.get_volume()
    assert imported.spatial_transform is not None
    assert imported.meta['intensity_transform'] == 'vol'
    np.testing.assert_array_equal(imported.raw_data, raw)
    np.testing.assert_array_equal(imported.layers['ILM'].raw_data, raw_heights)
    assert imported.layers['ILM'].manual == [True, False]
