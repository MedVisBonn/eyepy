from dataclasses import replace
import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from eyepy import EyeVolume
from eyepy.io import HeXmlExportContext
from eyepy.io import HeXmlReader
from eyepy.io import HeXmlWriter
from eyepy.io.he._xml_export import serialize_heyex_xml

SOURCE = Path('tests/data/filetypes/heyex_xml/test_volume')


@pytest.fixture
def volume():
    return HeXmlReader(SOURCE).volume


@pytest.fixture
def context():
    return HeXmlExportContext(
        pid=101, study_id=202, series_id=303,
        software_name='Synthetic Export', software_version='1.0',
        align_followup_to_reference=False,
        reference_id=303, reference_exam_date='2000-01-01',
        visit_date='2000-01-02', patient_id='new-patient-id',
        patient_uids=(('synthetic-source', 'synthetic-patient-uid'),),
        study_uid='synthetic-study-uid', series_uid='synthetic-series-uid',
        examined_structure='Retina', modality_procedure='IR_OCT', series_type='Volume',
        equipment={'Manufacturer': 'Example', 'ManufacturerModelName': 'Synthetic',
                   'AQMName': 'Synthetic Acquisition', 'AQMVersion': '1.2'},
        oct_field_size=(20, 15), utc_bias_minutes=120,
        filename='generated.xml', exam_url_prefix='file:///C:\\synthetic\\')


def test_generated_bundle_preserves_pixels_layers_flags_and_context(volume, context, tmp_path):
    pytest.importorskip('imagecodecs')
    original = volume.raw_data
    layer = volume.layers['ILM']
    layer.data[0, 10:14] = np.array([.100000001, 23.123456, np.nan, 37.125], np.float32)
    layer.manual = [None, True, False] + [None] * 7
    path = tmp_path / 'generated'
    HeXmlWriter(volume, export_context=context).write(path)
    assert len(list(path.iterdir())) == volume.size_z + 2
    reader = HeXmlReader(path)
    restored = reader.volume
    np.testing.assert_array_equal(restored.data, original)
    np.testing.assert_array_equal(volume.raw_data, original)
    np.testing.assert_array_equal(restored.localizer.data, volume.localizer.data)
    for name, layer in volume.layers.items():
        np.testing.assert_array_equal(restored.layers[name].data, layer.data.astype(np.float32))
        assert restored.layers[name].manual == layer.manual
        assert restored.layers[name].meta['heyex_xml_enabled'] == layer.meta['heyex_xml_enabled']
    assert reader.parsed_values['pid'] == 101
    assert reader.parsed_values['vid'] == 202
    assert restored.meta['id'] == restored.meta['reference_id'] == '303'
    assert restored.meta['patient']['patient_id'] == 'new-patient-id'
    assert restored.meta['reference_exam_date'] == '2000-01-01T00:00:00'
    assert restored.meta['exam_time'] is None
    assert restored.meta['visit_date'] == '2000-01-02T00:00:00'
    assert restored.meta['bscan_meta'][0]['heyex_xml_acquisition_time'] == '11:15:40.693000+02:00'
    assert restored.localizer.meta['heyex_xml_acquisition_time'] == '11:15:40.046000+02:00'
    assert restored.meta['bscan_meta'][0]['art_mode'] == 13
    raw = (path / 'generated.xml').read_bytes()
    assert not raw.startswith(b'<?xml')
    assert raw.endswith(b'\r\n') and b'\n' not in raw.replace(b'\r\n', b'')
    assert b'<Year>2000</Year>\r\n<Month>01</Month>' in raw


def test_segmentation_setting_is_explicit(volume, context):
    with_layers = ET.fromstring(HeXmlWriter(volume, export_context=context).bytes)
    without_layers = ET.fromstring(HeXmlWriter(volume, export_context=context, include_segmentation=False).bytes)
    assert len(with_layers.findall('.//SegLine')) == len(volume.layers) * volume.size_z
    assert not without_layers.findall('.//SegLine')
    np.testing.assert_array_equal(volume.layers['RPE'].manual, [True] * volume.size_z)


def test_unknown_manual_flags_do_not_mutate_source_metadata(volume, context):
    layer = volume.layers['ILM']
    layer.meta.pop('manual', None)
    root = ET.fromstring(HeXmlWriter(volume, export_context=context).bytes)
    assert 'manual' not in layer.meta
    assert root.find(".//SegLine[Name='ILM']/Manual") is None


def test_disabled_layer_status_is_preserved(context):
    reader = HeXmlReader(SOURCE)
    reader.xml_root.find(".//SegLine[Name='ILM']/Enabled").text = 'false'
    volume = reader.volume
    assert volume.layers['ILM'].meta['heyex_xml_enabled'][0] is False
    root = ET.fromstring(HeXmlWriter(volume, export_context=context).bytes)
    assert root.findtext(".//SegLine[Name='ILM']/Enabled") == 'false'


def test_generated_bundle_is_atomic_and_does_not_clobber(volume, context, tmp_path):
    pytest.importorskip('imagecodecs')
    path = tmp_path / 'generated'
    path.mkdir()
    (path / 'keep').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        HeXmlWriter(volume, export_context=context).write(path)
    assert (path / 'keep').read_bytes() == b'keep'
    # Valid document metadata followed by invalid display data must leave
    # neither a partial published bundle nor a temporary directory.
    volume.set_intensity_transform(lambda values: values.astype(float))
    failed = tmp_path / 'failed'
    with pytest.raises(ValueError, match='uint8'):
        HeXmlWriter(volume, export_context=context).write(failed)
    assert not failed.exists()
    assert not list(tmp_path.glob('.failed.*'))


def test_context_and_image_edits_are_distinct_from_native_rewriting(volume, context):
    with pytest.raises(TypeError, match='HeXmlExportContext'):
        HeXmlWriter(volume)
    with pytest.raises(ValueError, match='reader.xml_root'):
        HeXmlWriter(HeXmlReader(SOURCE), export_context=context)
    with pytest.raises(ValueError, match='source EyeVolume'):
        HeXmlWriter(volume, export_context=context).replace_image('localizer.tif', volume.localizer.data)


@pytest.mark.parametrize('name', ['../escape.xml', 'bad\\name.xml', 'bad\nname.xml', 'bad.tif'])
def test_generated_filenames_are_plain(volume, context, name):
    with pytest.raises(ValueError, match='plain filenames'):
        HeXmlWriter(volume, export_context=replace(context, filename=name))


def test_asset_names_and_count_are_validated(volume, context):
    with pytest.raises(ValueError, match='one localizer and every B-scan'):
        HeXmlWriter(volume, export_context=replace(context, image_names=('only.tif',)))
    with pytest.raises(ValueError, match='distinct'):
        HeXmlWriter(volume, export_context=replace(context, image_names=('same.tif',) * 11))
    names = ('same.tif', 'SAME.TIF') + tuple(f'{i}.tif' for i in range(9))
    with pytest.raises(ValueError, match='distinct'):
        HeXmlWriter(volume, export_context=replace(context, image_names=names))


def test_external_reference_requires_context_and_rejects_unknown_alignment(volume, context):
    volume.meta['e2e_acquisition_key'] = {'progression_id': 'example', 'windows_ticks': 2}
    volume.meta['e2e_reference_key'] = {'progression_id': 'example', 'windows_ticks': 1}
    with pytest.raises(ValueError, match='requires a receiving-database'):
        HeXmlWriter(volume, export_context=replace(context, reference_id=None)).bytes
    with pytest.raises(ValueError, match='cannot use the acquisition'):
        HeXmlWriter(volume, export_context=context).bytes
    with pytest.raises(NotImplementedError, match='not yet reproduced'):
        HeXmlWriter(volume, export_context=replace(context, reference_id=302,
                                                  align_followup_to_reference=True)).bytes
    root = ET.fromstring(HeXmlWriter(volume, export_context=replace(context, reference_id=302)).bytes)
    assert root.findtext('.//ReferenceSeries/ID') == '302'


def test_self_reference_uses_new_database_id(volume, context):
    volume.meta['e2e_acquisition_key'] = volume.meta['e2e_reference_key'] = {'example': 1}
    root = ET.fromstring(HeXmlWriter(volume, export_context=replace(
        context, reference_id=None, align_followup_to_reference=True)).bytes)
    assert root.findtext('.//ReferenceSeries/ID') == '303'


@pytest.mark.parametrize('clock,bias,hour,minute,second,expected_bias', [
    ('2000-01-01T12:34:56.123+02:00', None, '12', '34', '56.123', '120'),
    ('2000-01-01T23:59:59.999600', 120, '0', '0', '0.000', '120'),
    (datetime.datetime(2000, 7, 1, 12, 34, tzinfo=ZoneInfo('Europe/Berlin')), None, '12', '34', '0.000', '120'),
])
def test_clock_offset_precision_and_carry(volume, context, clock, bias, hour, minute, second, expected_bias):
    # Remove the other naive source clocks when testing aware standalone ones.
    for meta in volume.meta['bscan_meta']:
        meta.pop('heyex_xml_acquisition_time', None)
    root = ET.fromstring(HeXmlWriter(volume, export_context=replace(
        context, localizer_time=clock, utc_bias_minutes=bias)).bytes)
    node = root.find('.//Image/AcquisitionTime/Time')
    assert [node.findtext(key) for key in ('Hour', 'Minute', 'Second', 'UTCBias')] == [hour, minute, second, expected_bias]


def test_naive_clock_requires_offset_and_aware_clock_cannot_conflict(volume, context):
    with pytest.raises(ValueError, match='Naive acquisition times'):
        HeXmlWriter(volume, export_context=replace(context, utc_bias_minutes=None, localizer_time='12:00:00')).bytes
    with pytest.raises(ValueError, match='disagrees'):
        HeXmlWriter(volume, export_context=replace(context, localizer_time='12:00:00+01:00')).bytes


def test_missing_dates_are_not_invented(volume, context):
    volume.meta['visit_date'] = None
    with pytest.raises(ValueError, match='visit_date is required'):
        HeXmlWriter(volume, export_context=replace(context, visit_date=None)).bytes
    with pytest.raises(ValueError, match='reference_exam_date is required'):
        HeXmlWriter(volume, export_context=replace(context, reference_id=302, reference_exam_date=None)).bytes
    volume.meta['patient']['dob'] = None
    with pytest.raises(ValueError, match='birth date is required'):
        HeXmlWriter(volume, export_context=context).bytes


@pytest.mark.parametrize('changes,message', [
    ({'pid': None}, 'database IDs'), ({'series_id': True}, 'database IDs'),
    ({'software_version': ''}, 'software name and version'),
    ({'scale_precision': 0}, 'rounds a pixel scale to zero'),
    ({'scale_precision': 16}, 'numeric precision'),
    ({'position_precision': -1}, 'numeric precision'),
    ({'align_followup_to_reference': 1}, 'boolean XML alignment'),
    ({'localizer_acquisition': {'Focus': float('nan')}}, 'must be finite'),
    ({'oct_acquisition': {'Width': 1}}, 'Unsupported XML acquisition'),
    ({'equipment': {'unknown': 'value'}}, 'Unsupported XML equipment'),
])
def test_invalid_context_cannot_publish_metadata(volume, context, changes, message):
    with pytest.raises(ValueError, match=message):
        HeXmlWriter(volume, export_context=replace(context, **changes)).bytes


def test_serialized_profile_preserves_scalar_text_and_attributes():
    root = ET.Element('HEDX')
    node = ET.SubElement(root, 'Value', {'n': 'quote"&\t\n\r'})
    node.text = 'Müller < & > "\rnext'
    encoded = serialize_heyex_xml(root)
    restored = ET.fromstring(encoded).find('Value')
    assert restored.text == node.text
    assert restored.attrib == node.attrib
    assert b' />' not in encoded
    assert b'\r\n' in encoded


@pytest.mark.parametrize('kind', ['mixed', 'tail', 'comment', 'name', 'control'])
def test_generated_profile_rejects_unsupported_xml(kind):
    root = ET.Element('HEDX')
    child = ET.SubElement(root, 'Child')
    if kind == 'mixed':
        root.text = 'mixed text'
    elif kind == 'tail':
        child.tail = 'mixed tail'
    elif kind == 'comment':
        root.append(ET.Comment('comment'))
    elif kind == 'name':
        child.tag = 'bad name'
    else:
        child.text = 'invalid\x00text'
    with pytest.raises(ValueError):
        serialize_heyex_xml(root)


def test_xml_clock_and_reference_date_survive_eye_without_claiming_exam_date(volume, tmp_path):
    assert volume.meta['exam_time'] is None
    assert volume.meta['reference_exam_date'] == '2000-01-01T00:00:00'
    archive = tmp_path / 'clock.eye'
    volume.save(archive)
    restored = EyeVolume.load(archive)
    assert restored.meta == volume.meta
    assert restored.localizer.meta.as_dict() == volume.localizer.meta.as_dict()
