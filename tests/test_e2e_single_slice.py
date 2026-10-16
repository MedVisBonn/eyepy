import struct
from types import SimpleNamespace

import numpy as np
import pytest

from eyepy import EyeVolume
from eyepy.io.he.e2e_format import LateralityEnum
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.he.e2e_reader import E2ESeriesStructure
from eyepy.io.he.e2e_reader import E2ESliceStructure


@pytest.fixture(params=[0, 2], ids=['slice-id-0', 'slice-id-2'])
def single_slice_series(request):
    series = E2ESeriesStructure(1)
    series.folders[TypesEnum.laterality] = [
        SimpleNamespace(data=SimpleNamespace(laterality=LateralityEnum.OD))
    ]
    series.folders[TypesEnum.enface_modality] = [
        SimpleNamespace(data=SimpleNamespace(text=['', 'IR']))
    ]
    raw = bytearray(438)
    struct.pack_into('<III', raw, 0, 0, 4, 16)
    struct.pack_into('<4f', raw, 12, -15, 0, 15, 0)
    struct.pack_into('<f', raw, 36, 0.00387)
    # The single-line export stores n_bscans=0 despite containing one image.
    struct.pack_into('<IIII', raw, 60, 16, 0, 0, 1)
    sl = E2ESliceStructure(request.param)
    sl.folders[TypesEnum.localizer_info] = [
        SimpleNamespace(data=SimpleNamespace(field_deg=30))
    ]
    sl.folders[TypesEnum.bscanmeta] = [
        SimpleNamespace(get_bytes=lambda: bytes(raw))
    ]
    sl.folders[TypesEnum.image] = [
        SimpleNamespace(data=SimpleNamespace(
            type=35652097, width=16, height=4,
            data=np.arange(64, dtype=np.float32).reshape(4, 16))),
        SimpleNamespace(data=SimpleNamespace(
            type=33620481, width=64, height=32,
            data=np.arange(2048, dtype=np.uint8).reshape(32, 64))),
    ]
    layer = np.linspace(1, 3, 16, dtype=np.float32)
    layer[-1] = 3.4e38
    sl.folders[TypesEnum.layer_annotation] = [
        SimpleNamespace(data=SimpleNamespace(id=0, data=layer, manual_edit_raw=1)),
        SimpleNamespace(data=SimpleNamespace(id=1, data=layer + 1, manual_edit_raw=0)),
    ]
    series.substructure[sl.id] = sl
    return series


@pytest.mark.parametrize('tilted', [False, True], ids=['horizontal', 'tilted'])
def test_single_slice_volume_preserves_images_layers_and_alignment(
        single_slice_series, tilted, tmp_path):
    series = single_slice_series
    sl = next(iter(series.slices.values()))
    if tilted:
        raw = bytearray(sl.folders[TypesEnum.bscanmeta][0].get_bytes())
        struct.pack_into('<4f', raw, 12, -10, -5, 10, 5)
        sl.folders[TypesEnum.bscanmeta] = [
            SimpleNamespace(get_bytes=lambda: bytes(raw))
        ]

    with pytest.warns(UserWarning, match='approximate optical model'):
        volume = series.get_volume()

    assert volume.shape == (1, 4, 16)
    assert volume.scale_unit == volume.localizer.scale_unit == 'mm'
    assert volume.scale_z == 0
    assert len(volume.meta['bscan_meta']) == 1
    np.testing.assert_array_equal(volume._raw_data[0], sl.get_bscan())
    np.testing.assert_array_equal(volume.localizer.data, sl.get_localizer())
    for name, layer_id, manual in [('ILM', 0, True), ('BM', 1, False)]:
        expected_layer = sl.get_layers()[layer_id].copy()
        expected_layer[-1] = np.nan
        np.testing.assert_array_equal(volume.layers[name].data, expected_layer[None, :])
        assert volume.layers[name].manual == [manual]

    meta = series.get_bscan_meta()[0]
    expected = (np.array([meta['start_pos'], meta['end_pos']]) + 15) / [30 / 64, 30 / 32]
    endpoints = np.array([[0, 0], [15, 0]])
    np.testing.assert_allclose(volume.localizer_transform(endpoints), expected, atol=1e-12)
    np.testing.assert_allclose(volume.localizer_transform.inverse(expected), endpoints, atol=1e-12)
    assert np.isfinite(volume.localizer_transform.params).all()

    path = tmp_path / 'single.eye'
    volume.save(path)
    restored = EyeVolume.load(path)
    np.testing.assert_array_equal(restored._raw_data, volume._raw_data)
    np.testing.assert_array_equal(restored.localizer.data, volume.localizer.data)
    np.testing.assert_allclose(restored.localizer_transform.params, volume.localizer_transform.params)
    for name in volume.layers:
        np.testing.assert_array_equal(restored.layers[name].data, volume.layers[name].data)
        assert restored.layers[name].manual == volume.layers[name].manual
