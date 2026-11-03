import numpy as np
import pytest

import eyepy as ep
from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.io import HeVolReader
from eyepy.io import HeVolWriter


@pytest.fixture
def vol_volume():
    return HeVolReader('tests/data/filetypes/heyex_vol/test_volume.vol').volume


def test_hevolreader():
    reader = HeVolReader('tests/data/filetypes/heyex_vol/test_volume.vol')
    assert len(reader.meta) > 0
    assert reader.meta['intensity_encoding'] == 'vol_linear'


def test_heyex_vol_import(vol_volume):
    assert vol_volume.shape == (10, 40, 50)
    assert vol_volume.localizer.shape == (50, 50)
    assert vol_volume.laterality == 'OD'
    assert vol_volume.intensity_encoding == 'vol_linear'
    assert vol_volume.linear_data_available


def test_vol_export_preserves_float32_bits_and_layers(vol_volume, tmp_path):
    volume = vol_volume
    # Include values that collapse into the same display bin, special values,
    # negative zero and a NaN payload. Export must preserve every stored bit.
    volume._raw_data[0, 0, :9] = [
        .1, .1001, .5, np.finfo(np.float32).max, np.inf, -np.inf, -0., 0., 1.]
    volume._raw_data.view('<u4')[0, 0, 9] = 0x7fc12345
    volume.set_intensity_transform(lambda values: np.zeros(values.shape, np.uint8))
    path = tmp_path / 'linear.vol'
    HeVolWriter(volume).write(path)
    loaded = HeVolReader(path).volume
    np.testing.assert_array_equal(loaded.raw_data.view('<u4'),
                                  volume.raw_data.view('<u4'))
    np.testing.assert_array_equal(loaded.localizer.raw_data,
                                  volume.localizer.data)
    assert loaded.meta['exam_time'] == volume.meta['exam_time']
    assert loaded.meta['visit_date'] == volume.meta['visit_date']
    assert loaded.meta['patient'] == volume.meta['patient']
    for name, layer in volume.layers.items():
        np.testing.assert_array_equal(loaded.layers[name].data,
                                      layer.data.astype(np.float32))
    # Missing layers must remain missing rather than becoming zero-height traces.
    assert np.isnan(loaded.layers['OPL'].data).all()


def test_e2e_export_writes_registered_linear_values(vol_volume, tmp_path):
    matrices = [[[1, 0, .5], [0, 1, 0]]] * vol_volume.size_z
    raw = np.full(vol_volume.shape, .1001, np.float32)
    raw[0, 0, :2] = [.75, 0]
    volume = ep.EyeVolume(
        raw, vol_volume.meta.copy(), vol_volume.localizer,
        vol_volume.localizer_transform,
        bscan_transform=HeidelbergBscanTransform([2] * len(raw), matrices),
        intensity_encoding='e2e_raw')
    expected = volume.linear_data
    assert expected[0, 0, 0] == .5  # Clip before interpolation.
    assert expected[0, 0, -1] == np.finfo(np.float32).max
    volume.set_intensity_transform(lambda values: np.zeros(values.shape, np.uint8))
    path = tmp_path / 'registered.vol'
    HeVolWriter(volume).write(path)
    loaded = HeVolReader(path).volume
    np.testing.assert_array_equal(loaded.raw_data.view('<u4'),
                                  expected.view('<u4'))
    np.testing.assert_array_equal(volume.raw_data, raw)
    assert np.isnan(loaded.layers['ILM'].data).all()


def test_xml_export_requires_explicit_approximation_and_preserves_greys(tmp_path):
    volume = ep.import_heyex_xml('tests/data/filetypes/heyex_xml/test_volume')
    assert volume.intensity_encoding == 'display'
    assert not volume.linear_data_available
    original = volume.raw_data
    for data in (volume, volume[0]):
        with pytest.raises(ValueError, match='VOL-linear data is unavailable'):
            data.linear_data
    path = tmp_path / 'display.vol'
    path.write_bytes(b'existing file')
    with pytest.raises(ValueError, match='allow_approximate=True'):
        HeVolWriter(volume).write(path)
    assert path.read_bytes() == b'existing file'
    # Export uses stored display greys, not a user-selected visualization.
    volume.set_intensity_transform(lambda values: np.zeros(values.shape, np.uint8))
    HeVolWriter(volume, allow_approximate=True).write(path)
    loaded = HeVolReader(path).volume
    np.testing.assert_array_equal(loaded.data, original)
    np.testing.assert_array_equal(volume.raw_data, original)


@pytest.mark.parametrize('encoding', ['unknown', 'e2e_raw'])
@pytest.mark.parametrize('allow_approximate', [False, True])
def test_export_rejects_unknown_or_unnormalized_data(
        vol_volume, encoding, allow_approximate):
    volume = ep.EyeVolume(
        vol_volume.raw_data, vol_volume.meta.copy(), vol_volume.localizer,
        vol_volume.localizer_transform, intensity_encoding=encoding)
    with pytest.raises(ValueError, match='VOL-linear data is unavailable'):
        HeVolWriter(volume, allow_approximate=allow_approximate).bytes
