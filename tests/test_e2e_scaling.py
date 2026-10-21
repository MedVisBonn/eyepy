from io import BytesIO
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from eyepy.io.he.e2e_format import LateralityEnum
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.he.e2e_reader import E2EFolder
from eyepy.io.he.e2e_reader import E2ESeriesStructure
from eyepy.io.he.e2e_reader import E2ESliceStructure
from eyepy.io.he.e2e_reader import HeE2eReader


def localizer_info(field_deg):
    raw = bytearray(59)
    struct.pack_into('<H', raw, 2, ord('R'))
    struct.pack_into('<2H', raw, 12, 64, 32)
    struct.pack_into('<H', raw, 24, field_deg)
    raw[26:28] = b'\xff\xff'  # The field is two bytes, not four.
    return bytes(raw)


def record_folder(type_id, raw):
    """A real folder backed by an in-memory E2E container."""
    header = struct.pack('<12sIIIIIiiiiHHII', b'CONTAINER', 0, 0, 0,
                         len(raw), 0, 1, 1, 1, 0, 0, 0, type_id, 0)
    reader = SimpleNamespace(file_object=BytesIO(header + raw))
    return E2EFolder(1, 1, 1, 0, 0, 0, type_id, len(raw), 0, reader)


@pytest.fixture
def series():
    """Two 20-degree B-scans with incomplete filled-A-scan coverage."""
    result = E2ESeriesStructure(1)
    result.folders[TypesEnum.laterality] = [
        SimpleNamespace(data=SimpleNamespace(laterality=LateralityEnum.OD))
    ]
    result.folders[TypesEnum.enface_modality] = [
        SimpleNamespace(data=SimpleNamespace(text=['', 'IR']))
    ]
    result.folders[TypesEnum.image] = [SimpleNamespace(data=SimpleNamespace(
        type=33620481, width=64, height=32, data=np.zeros((32, 64))))]
    settings = bytearray(600)
    struct.pack_into('<d', settings, 457, 7.7)
    for index, y in enumerate((-1, 1)):
        raw = bytearray(438)
        struct.pack_into('<III', raw, 0, 0, 4, 16)
        struct.pack_into('<4f', raw, 12, -10, y, 10, y)
        struct.pack_into('<f', raw, 36, 0.00387)
        struct.pack_into('<IIII', raw, 60, 12, 2, index, 3)
        struct.pack_into('<2f', raw, 76, 0, y)
        struct.pack_into('<d', raw, 136, 0.9 / 0.001043)
        sl = E2ESliceStructure(index * 2)
        sl.folders[TypesEnum.bscanmeta] = [
            SimpleNamespace(get_bytes=lambda raw=raw: bytes(raw))
        ]
        sl.folders[TypesEnum.localizer_settings] = [
            record_folder(39, bytes(settings))
        ]
        sl.folders[5] = [record_folder(5, localizer_info(30))]
        sl.folders[TypesEnum.layer_annotation] = []
        sl.get_bscan = lambda: np.zeros((4, 16))
        result.substructure[sl.id] = sl
    return result


def test_default_volume_uses_consistent_mm_and_preserves_alignment(series):
    q = 0.2895099745331661  # R=7.7 mm, F=0.9 D, independently evaluated.
    angular = series.get_bscan_meta()

    with pytest.warns(UserWarning, match='approximate optical model'):
        volume = series.get_volume()

    assert volume.scale_unit == volume.localizer.scale_unit == 'mm'
    assert volume.scale == pytest.approx((2 * q, 0.00387, 20 * q / 16))
    assert volume.localizer.scale_x == pytest.approx(30 * q / 64)
    assert volume.localizer.scale_y == pytest.approx(30 * q / 32)
    assert angular[0]['pos_unit'] == '°'
    assert angular[0]['start_pos'] == (-10, -1)
    assert volume.meta['bscan_meta'][0]['start_pos'] == pytest.approx((5*q, 14*q))
    assert volume.meta['bscan_meta'][0]['pos_unit'] == 'mm'
    # OCT row 0 maps to the final B-scan in the localizer, as before conversion.
    expected = np.array([[5 / 30 * 64, 16 / 30 * 32],
                         [25 / 30 * 64, 14 / 30 * 32]])
    np.testing.assert_allclose(volume.localizer_transform([[0, 0], [15, 1]]), expected)
    info = volume.meta['e2e_metadata']['findings']['scaling_x']
    assert info['status'] == 'approximate'
    assert info['image_width_px'] == 16  # Not the 12 filled A-scans.
    assert info['corneal_radius_mm'] == 7.7
    assert info['focus_diopters'] == pytest.approx(0.9)


@pytest.mark.parametrize('field_deg', [20, 55])
def test_stored_localizer_field_controls_scale_origin_and_alignment(series, field_deg):
    for sl in series.slices.values():
        sl.folders[5] = [record_folder(5, localizer_info(field_deg))]
    q = 0.2895099745331661
    half_field = field_deg / 2

    with pytest.warns(UserWarning, match='approximate optical model'):
        volume = series.get_volume()

    assert series.get_localizer_field_deg() == field_deg
    assert volume.scale_x == pytest.approx(20 * q / 16)
    assert volume.localizer.scale_x == pytest.approx(field_deg * q / 64)
    assert volume.localizer.scale_y == pytest.approx(field_deg * q / 32)
    assert volume.localizer.meta['field_size'] == field_deg
    assert volume.meta['bscan_meta'][0]['start_pos'] == pytest.approx(
        ((half_field - 10) * q, (half_field - 1) * q))
    expected = np.array([
        [(half_field - 10) / field_deg * 64, (half_field + 1) / field_deg * 32],
        [(half_field + 10) / field_deg * 64, (half_field - 1) / field_deg * 32],
    ])
    np.testing.assert_allclose(volume.localizer_transform([[0, 0], [15, 1]]), expected, atol=1e-12)
    assert volume.meta['e2e_metadata']['findings']['localizer_field'] == {
        'field_deg': field_deg, 'source': 'Type5@offset24:uint16'}
    assert series.get_heyex_metadata()['IR Image']['Scan Angle'] == f'{field_deg}°'
    assert series.get_heyex_metadata_sources()['IR Image']['Scan Angle'] == 'Type5@offset24:uint16'


def test_localizer_field_can_be_stored_at_series_level(series):
    series.folders[5] = [record_folder(5, localizer_info(55))]
    assert series.get_localizer_field_deg() == 55


@pytest.mark.parametrize(('raw', 'message'), [
    (None, 'no Type5 record'),
    (localizer_info(30)[:25], 'payload has 25 bytes; expected at least 26'),
    (localizer_info(0), 'offset 24 is 0 degrees'),
    (localizer_info(180), 'offset 24 is 180 degrees'),
    (localizer_info(286), 'offset 24 is 286 degrees'),
    (localizer_info(65535), 'offset 24 is 65535 degrees'),
])
def test_missing_short_or_invalid_localizer_field_raises(series, raw, message):
    for sl in series.slices.values():
        sl.folders[5] = [] if raw is None else [record_folder(5, raw)]

    with pytest.raises(ValueError, match=message):
        series.get_localizer_field_deg()

    with pytest.raises(ValueError, match=message):
        series.get_volume()


def test_invalid_series_field_is_not_masked_by_valid_slice_field(series):
    series.folders[5] = [record_folder(5, localizer_info(0))]
    with pytest.raises(ValueError, match='offset 24 is 0 degrees'):
        series.get_localizer_field_deg()


@pytest.mark.parametrize('attribute', ['volume', 'volumes'])
def test_reader_preserves_localizer_field_error_after_empty_series(
        series, monkeypatch, attribute):
    for sl in series.slices.values():
        sl.folders[5] = []
    reader = HeE2eReader.__new__(HeE2eReader)
    monkeypatch.setattr(HeE2eReader, 'series',
                        property(lambda self: [E2ESeriesStructure(0), series]))

    with pytest.raises(ValueError, match='no Type5 record'):
        getattr(reader, attribute)


@pytest.mark.parametrize(('radius', 'focus', 'expected_q'), [
    (9.0, -1.51, 0.3329369347634005),
    (6.2, -0.49, 0.2560921282567403),
])
def test_optical_conversion_matches_controlled_radius_examples(
        series, monkeypatch, radius, focus, expected_q):
    monkeypatch.setattr(series, 'get_corneal_radius_mm', lambda: radius)
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: focus)
    assert series._mm_per_degree() == pytest.approx(expected_q)


@pytest.mark.parametrize('focus', [None, float('nan')])
def test_missing_optical_settings_use_documented_defaults(series, monkeypatch, focus):
    monkeypatch.setattr(series, 'get_corneal_radius_mm', lambda: None)
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: focus)
    assert series._mm_per_degree() == pytest.approx(1.02302 / (17.21 / 7.7 + 1.247))


@pytest.mark.parametrize('eye,expected_radius', [
    (LateralityEnum.OD, 6.2),
    (LateralityEnum.OS, 9.0),
])
def test_study_radius_fallback_selects_the_matching_eye(series, eye, expected_radius):
    for sl in series.slices.values():
        sl.folders[TypesEnum.localizer_settings] = [record_folder(39, bytes(464))]
    series.folders[TypesEnum.laterality][0].data.laterality = eye
    series.study = SimpleNamespace(folders={TypesEnum.measurements: [
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OS, corneal_radius_mm=9.0)),
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OD, corneal_radius_mm=6.2)),
    ]})
    assert series.get_corneal_radius_mm() == expected_radius


def test_study_and_all_slices_must_have_the_same_radius(series):
    series.study = SimpleNamespace(folders={TypesEnum.measurements: [
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OS, corneal_radius_mm=9.0)),
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OD, corneal_radius_mm=7.7)),
    ]})
    assert series.get_corneal_radius_mm() == 7.7
    series.study.folders[TypesEnum.measurements][1].data.corneal_radius_mm = 9.0
    with pytest.raises(ValueError, match='Inconsistent E2E corneal radii'):
        series.get_corneal_radius_mm()


@pytest.mark.parametrize(('radius', 'message'), [
    (9.0, 'Inconsistent E2E corneal radii'),
    (0.0, 'Invalid E2E corneal radius'),
    (-1.0, 'Invalid E2E corneal radius'),
    (float('nan'), 'Invalid E2E corneal radius'),
    (float('inf'), 'Invalid E2E corneal radius'),
])
def test_invalid_or_inconsistent_slice_radius_is_rejected(series, radius, message):
    raw = bytearray(600)
    struct.pack_into('<d', raw, 457, radius)
    series.slices[2].folders[TypesEnum.localizer_settings] = [record_folder(39, raw)]
    with pytest.raises(ValueError, match=message):
        series.get_corneal_radius_mm()


@pytest.mark.parametrize('attribute', ['volume', 'volumes'])
@pytest.mark.parametrize('failure', ['denominator', 'radius', 'geometry'])
def test_reader_preserves_scaling_errors(series, monkeypatch, attribute, failure):
    if failure == 'denominator':
        monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: -100)
        message = 'non-positive scale'
    elif failure == 'radius':
        series.study = SimpleNamespace(folders={TypesEnum.measurements: [
            SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OD, corneal_radius_mm=9.0)),
        ]})
        message = 'Inconsistent E2E corneal radii'
    else:
        series._bscanmeta_item(0).end_x = -10
        message = 'Invalid E2E B-scan geometry'
    reader = HeE2eReader.__new__(HeE2eReader)
    monkeypatch.setattr(HeE2eReader, 'series',
                        property(lambda self: [E2ESeriesStructure(0), series]))
    with pytest.warns(UserWarning), pytest.raises(ValueError, match=message):
        getattr(reader, attribute)


def test_single_bscan_has_no_measured_through_slice_spacing(series):
    del series.substructure[2]
    with pytest.warns(UserWarning):
        meta = series.get_meta()
    assert meta['scale_z'] == 0


def test_invalid_optical_denominator_is_rejected(series, monkeypatch):
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: -100)
    with pytest.raises(ValueError, match='non-positive scale'):
        series._mm_per_degree()
