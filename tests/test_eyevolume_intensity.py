import json
import zipfile

import numpy as np
import pytest

from eyepy import EyeVolume
from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.io import HeVolReader


def volume_with_encoding(encoding):
    source = HeVolReader('tests/data/filetypes/heyex_vol/test_volume.vol').volume
    return EyeVolume(np.full(source.shape, .5, np.float32), source.meta.copy(),
                     source.localizer, source.localizer_transform,
                     intensity_encoding=encoding)


def rewrite_metadata(source, target, update):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, 'w') as output:
        for name in original.namelist():
            content = original.read(name)
            if name == 'meta.json':
                meta = json.loads(content)
                update(meta)
                content = json.dumps(meta).encode()
            output.writestr(name, content)


@pytest.mark.parametrize('encoding', ['unknown', 'display', 'e2e_raw', 'vol_linear'])
@pytest.mark.parametrize('display', ['default', 'vol', 'e2e'])
def test_encoding_survives_save_load_independently_of_display(
        encoding, display, tmp_path):
    volume = volume_with_encoding(encoding)
    volume.set_intensity_transform(display)
    path = tmp_path / 'encoded.eye'
    volume.save(path)
    loaded = EyeVolume.load(path)
    assert loaded.intensity_encoding == encoding
    assert loaded.meta['intensity_transform'] == display
    np.testing.assert_array_equal(loaded.raw_data, volume.raw_data)
    np.testing.assert_array_equal(loaded.data, volume.data)
    np.testing.assert_array_equal(loaded[0].data, volume[0].data)
    if encoding == 'vol_linear':
        np.testing.assert_array_equal(loaded.linear_data, volume.raw_data)
        np.testing.assert_array_equal(loaded[0].linear_data, volume.raw_data[0])
    else:
        for data in (loaded, loaded[0]):
            with pytest.raises(ValueError, match='VOL-linear data is unavailable'):
                data.linear_data


def test_new_arrays_do_not_infer_encoding_from_dtype_range_or_display():
    # The same float samples and display function can represent different domains.
    for dtype in (np.uint8, np.float32):
        volume = EyeVolume(np.zeros((2, 3, 4), dtype))
        volume.set_intensity_transform('vol')
        assert volume.intensity_encoding == 'unknown'
        assert not volume.linear_data_available
    with pytest.raises(ValueError, match='Unknown intensity encoding'):
        EyeVolume(np.zeros((2, 3, 4)), intensity_encoding='typo')


@pytest.mark.parametrize('display,expected', [
    ('vol', 'vol_linear'), ('e2e', 'e2e_raw'), ('default', 'unknown')])
def test_older_files_use_saved_intensity_function(display, expected, tmp_path):
    volume = volume_with_encoding('vol_linear')
    volume._raw_data[0, 0, :3] = [.25, .5, 1.9990234375]
    volume.set_intensity_transform(display)
    path = tmp_path / 'new.eye'
    old_path = tmp_path / 'legacy.eye'
    volume.save(path)
    rewrite_metadata(path, old_path, lambda meta: meta.pop('intensity_encoding'))
    loaded = EyeVolume.load(old_path)
    assert loaded.intensity_encoding == expected
    assert loaded.meta['intensity_transform'] == display
    np.testing.assert_array_equal(loaded.raw_data, volume.raw_data)
    np.testing.assert_array_equal(loaded.data, volume.data)
    np.testing.assert_array_equal(loaded[0].data, volume[0].data)
    assert loaded.linear_data_available == (expected == 'vol_linear')
    if expected == 'vol_linear':
        np.testing.assert_array_equal(loaded.linear_data, volume.raw_data)
    else:
        with pytest.raises(ValueError, match='VOL-linear data is unavailable'):
            loaded.linear_data


def test_older_custom_display_file_keeps_ambiguous_encoding(tmp_path):
    volume = volume_with_encoding('vol_linear')
    volume.set_intensity_transform(lambda values: values + 1)
    path = tmp_path / 'custom.eye'
    old_path = tmp_path / 'older.eye'
    with pytest.warns(UserWarning, match='Custom intensity transforms'):
        volume.save(path)
    rewrite_metadata(path, old_path, lambda meta: meta.pop('intensity_encoding'))
    loaded = EyeVolume.load(old_path)
    assert loaded.intensity_encoding == 'unknown'
    assert loaded.meta['intensity_transform'] == 'default'
    np.testing.assert_array_equal(loaded.data, volume.raw_data)
    assert not loaded.linear_data_available


def test_transform_rejects_unknown_versions_and_incompatible_encodings():
    source = volume_with_encoding('vol_linear')
    transform = HeidelbergBscanTransform(
        [1] * source.size_z, [[[1, 0, 0], [0, 1, 0]]] * source.size_z)
    for version in (None, 2):
        config = transform.as_dict()
        if version is None:
            config.pop('version')
        else:
            config['version'] = version
        with pytest.raises(ValueError, match='Unsupported B-scan transform version'):
            HeidelbergBscanTransform.from_dict(config)
    with pytest.raises(ValueError, match='requires e2e_raw'):
        EyeVolume(source.raw_data, source.meta.copy(), source.localizer,
                  source.localizer_transform, bscan_transform=transform)


def test_raw_e2e_fallback_displays_without_claiming_linear_conversion(monkeypatch):
    from eyepy.io.he.e2e_reader import E2ESeriesStructure

    source = volume_with_encoding('e2e_raw')
    source.meta['bscan_meta'][0]['scan_pattern'] = 3
    series = E2ESeriesStructure(1)
    monkeypatch.setattr(series, 'get_meta', lambda: source.meta)
    monkeypatch.setattr(series, 'get_bscans', lambda: source.raw_data)
    monkeypatch.setattr(series, 'get_localizer', lambda registered: source.localizer)
    monkeypatch.setattr(series, 'get_layers', lambda: {})
    monkeypatch.setattr(series, 'get_layer_manual_maps', lambda: {})
    with pytest.warns(RuntimeWarning, match='normalization/registration unavailable'):
        loaded = series.get_volume()
    assert loaded.intensity_encoding == 'e2e_raw'
    assert not loaded.linear_data_available
    assert loaded.data.shape == source.shape
    with pytest.raises(ValueError, match='VOL-linear data is unavailable'):
        loaded.linear_data
