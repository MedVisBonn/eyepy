import datetime
import struct

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.io import HeVolExportContext
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


def test_vol_export_uses_e2e_ui_quality_and_consistent_header_sizes(vol_volume, tmp_path):
    for meta in vol_volume.meta['bscan_meta']:
        meta['quality'] = .84  # Legacy E2E field is not the displayed quality.
        meta['quality_ui'] = 29.5
    path = tmp_path / 'quality.vol'
    HeVolWriter(vol_volume).write(path)
    parsed = HeVolReader(path).parsed_file
    assert all(bscan.quality == 29.5 for bscan in parsed.bscans)
    assert all(bscan.bscan_hdr_size == parsed.bscan_hdr_size for bscan in parsed.bscans)


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
    # An export without segmentation vectors must not acquire invented layers.
    assert not loaded.layers


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
    assert not loaded.layers


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
    HeVolWriter(volume, allow_approximate=True).write(path, overwrite=True)
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


def test_generated_vol_preserves_metadata_dates_and_registration_through_eye(vol_volume, tmp_path):
    source = tmp_path / 'metadata.vol'
    native = HeVolReader('tests/data/filetypes/heyex_vol/test_volume.vol')
    parsed = native.parsed_file
    parsed.id, parsed.reference_id, parsed.prog_id = '21', '20', 'example-progression'
    parsed.vid, parsed.visit_id, parsed.scan_pattern = 17, 'visit-label', 3
    parsed.bscans[0].shift = 2
    parsed.bscans[0].iv_transformation = [1., 0., .5, .125, 1., -.25]
    HeVolWriter(native).write(source)
    # Include sub-microsecond time bits and a float64 day value that datetime
    # cannot represent exactly. Metadata serialization must retain those bits.
    raw = bytearray(source.read_bytes())
    ticks = struct.unpack_from('<Q', raw, 88)[0] + 3
    struct.pack_into('<Q', raw, 88, ticks)
    days = np.nextafter(struct.unpack_from('<d', raw, 200)[0], np.inf)
    struct.pack_into('<d', raw, 200, days)
    source.write_bytes(raw)
    volume = HeVolReader(source).volume
    archive = tmp_path / 'metadata.eye'
    volume.save(archive)
    volume = ep.EyeVolume.load(archive)
    output = tmp_path / 'generated.vol'
    HeVolWriter(volume).write(output)
    loaded = HeVolReader(output).parsed_file
    assert loaded.version == 'HSF-OCT-103'
    assert loaded.bscans[0].version == 'HSF-BS-103'
    for name in ('id', 'reference_id', 'prog_id', 'vid', 'visit_id', 'scan_pattern'):
        assert loaded[name] == parsed[name]
    assert loaded.bscans[0].shift == 2
    np.testing.assert_array_equal(loaded.bscans[0].iv_transformation,
                                  parsed.bscans[0].iv_transformation)
    assert output.read_bytes()[88:96] == raw[88:96]
    assert output.read_bytes()[164:172] == raw[164:172]
    assert output.read_bytes()[200:208] == raw[200:208]
    np.testing.assert_array_equal(loaded.bscans[0].data, parsed.bscans[0].data)


def test_explicit_export_context_overrides_receiving_database_fields(vol_volume, tmp_path):
    original_id = vol_volume.meta['id']
    context = HeVolExportContext(pid=71, vid=72, id='73', reference_id='70',
                                 patient_id='example-id', visit_id='example-visit',
                                 prog_id='example-progression', scan_pattern=3,
                                 exam_time=datetime.datetime(2001, 2, 3, 4, 5, 6),
                                 visit_date='2001-02-03T02:01:00')
    output = tmp_path / 'context.vol'
    HeVolWriter(vol_volume, export_context=context).write(output)
    parsed = HeVolReader(output).parsed_file
    for name, value in context.as_dict().items():
        assert parsed[name] == (value.isoformat() if isinstance(value, datetime.datetime) else value)
    assert vol_volume.meta['id'] == original_id
    # Edited dates take precedence over retained source timestamp encodings.
    assert parsed.exam_time == '2001-02-03T04:05:06'


def test_e2e_generation_requires_receiving_ids_and_writes_source_matrices(vol_volume, tmp_path):
    matrices = np.tile([[1., 0., .5], [.125, 1., .25]], (vol_volume.size_z, 1, 1))
    volume = ep.EyeVolume(np.full(vol_volume.shape, .1, np.float32),
                          vol_volume.meta.copy(), vol_volume.localizer,
                          vol_volume.localizer_transform,
                          bscan_transform=HeidelbergBscanTransform([1] * vol_volume.size_z, matrices),
                          intensity_encoding='e2e_raw')
    del volume.meta['id']
    volume.meta['e2e_acquisition_key'] = dict(progression_id='example', windows_ticks=2)
    volume.meta['e2e_reference_key'] = dict(progression_id='example', windows_ticks=1)
    with pytest.raises(ValueError, match='receiving-database identifiers'):
        HeVolWriter(volume).write(tmp_path / 'missing.vol')
    context = HeVolExportContext(pid=1, vid=2, id='3', scan_pattern=3)
    with pytest.raises(ValueError, match='reference_id is required'):
        HeVolWriter(volume, export_context=context).bytes
    # A self-reference maps to the newly assigned scan ID automatically.
    volume.meta['e2e_reference_key'] = volume.meta['e2e_acquisition_key']
    output = tmp_path / 'e2e-generated.vol'
    HeVolWriter(volume, export_context=context).write(output)
    parsed = HeVolReader(output).parsed_file
    assert parsed.reference_id == '3'
    for index, bscan in enumerate(parsed.bscans):
        np.testing.assert_array_equal(bscan.iv_transformation, matrices[index].ravel())
        np.testing.assert_array_equal(bscan.data.view('<u4'), volume[index].linear_data.view('<u4'))


def test_generated_vol_uses_present_layer_slots_and_vendor_missing_word(vol_volume, tmp_path):
    # A late slot caught the old fixed 17-layer allocation's IndexError.
    height = np.full((vol_volume.size_z, vol_volume.size_x), 12.5)
    height[0, 0] = np.nan
    vol_volume.add_layer_annotation(height, name='IPL-')
    output = tmp_path / 'layers.vol'
    HeVolWriter(vol_volume).write(output)
    parsed = HeVolReader(output).parsed_file
    assert all(bscan.num_seg == 19 for bscan in parsed.bscans)
    assert parsed.bscans[0].layer_segmentations.view('<u4')[18, 0] == 0x7f7fffff
    assert np.all(parsed.bscans[0].layer_segmentations.view('<u4')[:18] == 0x7f7fffff)
    np.testing.assert_array_equal(HeVolReader(output).volume.layers['IPL-'].data, height)


def test_quality_ui_does_not_require_the_legacy_quality_field(vol_volume, tmp_path):
    for meta in vol_volume.meta['bscan_meta']:
        del meta['quality']
        meta['quality_ui'] = 28.25
    output = tmp_path / 'ui-quality.vol'
    HeVolWriter(vol_volume).write(output)
    assert all(bscan.quality == 28.25 for bscan in HeVolReader(output).parsed_file.bscans)


def test_export_context_can_supply_exact_native_date_scalars(vol_volume, tmp_path):
    ticks = 125911584001234579
    days = np.nextafter(36526.123456789, np.inf)
    context = HeVolExportContext(pid=1, vid=2, id='3', exam_time_ticks=ticks,
                                 visit_date_days=float(days))
    output = tmp_path / 'native-context-dates.vol'
    HeVolWriter(vol_volume, export_context=context).write(output)
    raw = output.read_bytes()
    assert raw[88:96] == struct.pack('<Q', ticks)
    assert raw[200:208] == struct.pack('<d', days)
    assert HeVolReader(output).parsed_file.exam_time == '2000-01-01T00:00:00.123457'


def test_explicit_context_datetime_replaces_even_the_same_displayed_source_time(vol_volume):
    stamp = '2000-01-01T00:00:00.123457'
    vol_volume.meta['exam_time'] = stamp
    vol_volume.meta['heyex_native_dates'].update(exam_time=stamp,
                                               exam_time_ticks=125911584001234579)
    context = HeVolExportContext(pid=1, vid=2, id='3', exam_time=stamp)
    raw = HeVolWriter(vol_volume, export_context=context).bytes
    assert raw[88:96] == struct.pack('<Q', 125911584001234570)
