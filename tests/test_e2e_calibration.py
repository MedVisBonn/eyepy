"""Synthetic, non-identifying E2E fixtures for decoding and calibration."""

from io import BytesIO
import struct
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from eyepy.io.he.e2e_calibration import E2EOpticalModel
from eyepy.io.he.e2e_calibration import EMPIRICAL_OPTICAL_MODEL
from eyepy.io.he.e2e_calibration import inverse_localizer_registration
from eyepy.io.he.e2e_calibration import RESEARCH_OPTICAL_MODEL
from eyepy.io.he.e2e_format import datacontainer_format
from eyepy.io.he.e2e_format import type10004_format
from eyepy.io.he.e2e_reader import E2EFolder
from eyepy.io.he.e2e_reader import E2ESeriesStructure
from eyepy.io.he.e2e_reader import E2ESliceStructure
from eyepy.io.he.e2e_reader import HeE2eReader

PROGRESSION = uuid.UUID('12345678-1234-5678-9abc-def012345678')
FOCUS_FACTOR = struct.unpack('<f', struct.pack('<f', 1.043))[0] / 1000


def container(typ, payload):
    header = struct.pack('<12sIIIIIiiiiHHII', b'MDbData', 0, 0, 0,
                         len(payload), 0, 1, 2, 3, -1, 1, 0, typ, 0)
    return header + payload


def bscan_payload(index=0, focus=.9, preset=0, endpoints=None):
    raw = bytearray(438)
    struct.pack_into('<2I', raw, 4, 4, 8)
    struct.pack_into('<4f', raw, 12, *(endpoints if endpoints is not None else (-15, 1-index, 15, 1-index)))
    struct.pack_into('<f', raw, 36, .00387)
    struct.pack_into('<4I', raw, 60, 8-index, 3, index, 1)
    struct.pack_into('<2f', raw, 76, 0, 1-index)
    struct.pack_into('<I', raw, 120, 7)
    struct.pack_into('<2d', raw, 136, focus/FOCUS_FACTOR, preset)
    struct.pack_into('<f', raw, 156, 35)
    raw[160:] = bytes([index + 1]) * 278
    return bytes(raw)


def make_series(focus=.9, radius=7.7, own_time=None, reference_time=None,
                affine=(1, 0, 0, 0, 1, 0), angles=None, missing_layers=False):
    series = E2ESeriesStructure(3)
    blob = bytearray(88)
    reader = SimpleNamespace(resolve_reference=lambda _: None)

    def folder(typ, payload, slice_id=-1):
        start = len(blob)
        blob.extend(container(typ, payload))
        return E2EFolder(1, 2, 3, slice_id, 0, start, typ, len(payload), 1, reader)

    series.folders[11] = [folder(11, bytes(14) + b'L')]
    series.folders[9007] = [SimpleNamespace(data=SimpleNamespace(text=['', 'IR']))]
    image = struct.pack('<5I', 276, 33620481, 256, 16, 16) + bytes(range(256))
    series.folders[1073741824] = [folder(1073741824, image)]
    eye = bytearray(68); eye[0] = ord('L'); struct.pack_into('<d', eye, 17, radius)
    series.study = SimpleNamespace(folders={7: [folder(7, eye)]})
    if own_time is not None:
        series.folders[10005] = [folder(10005, PROGRESSION.bytes_le + struct.pack('<Q', own_time))]
    if reference_time is not None:
        registration = (struct.pack('<2I', 3, 0x80000064) + PROGRESSION.bytes_le
                        + struct.pack('<Q6f', reference_time, *affine) + bytes(44))
        series.folders[10025] = [folder(10025, registration)]
    for i, sid in enumerate((42, 14, 98)):
        sl = E2ESliceStructure(sid)
        endpoints = None if angles is None else angles[i].ravel()
        sl.folders[10004] = [folder(10004, bscan_payload(i, focus, endpoints=endpoints), sid)]
        settings = bytearray(465); struct.pack_into('<d', settings, 457, radius)
        sl.folders[39] = [folder(39, settings, sid)]
        pixels = np.full((4, 8), 100 + i, dtype='<u2').tobytes()
        sl.folders[1073741824] = [folder(1073741824, struct.pack('<5I', 84, 35652097, 32, 4, 8) + pixels, sid)]
        if not missing_layers:
            layer = struct.pack('<9I', 2, 0, 0, 8, i % 2, 0, 0, 0, 0)
            layer += np.full(8, 1 + i, dtype='<f4').tobytes()
            sl.folders[10019] = [folder(10019, layer, sid)]
        series.slices[sid] = sl
    reader.file_object = BytesIO(blob)
    return series


@pytest.mark.parametrize('focus,preset,label', [(.9, 0, 'Medium'), (-1.51, -2500, 'Short'),
                                               (-.49, 2500, 'Long'), (-.06, 5000, 'Extra Long')])
def test_focus_double_offsets_preserve_following_fields(focus, preset, label):
    raw = bscan_payload(focus=focus, preset=preset)
    meta = type10004_format.parse(raw)
    assert meta.focus_raw == struct.unpack_from('<d', raw, 136)[0]
    assert meta.focus_diopters == pytest.approx(focus, abs=3e-16)
    assert meta.eye_length_preset == label
    assert meta.quality_ui == 35
    assert meta.rest == raw[160:]
    assert type10004_format.build(meta) == raw


def test_container_parse_cannot_consume_neighbor_payload():
    series = make_series()
    folder = series._sorted_slices()[0].folders[10004][0]
    assert len(folder.data.rest) == 278
    assert len(folder.get_bytes()) == 438
    assert folder.data is folder.data
    assert folder.header.size == 438


def test_truncated_and_inline_payloads_are_explicit():
    series = make_series()
    folder = series.slices[42].folders[10004][0]
    folder.reader.file_object = BytesIO(bytes(88) + container(10004, bscan_payload())[:-1])
    folder.start = 88
    with pytest.raises(EOFError):
        folder.get_bytes()
    folder.start = 1
    assert folder.is_inline
    with pytest.raises(ValueError, match='Inline'):
        folder.get_bytes()


@pytest.mark.parametrize('radius', [6.2, 7.7, 9.0])
def test_radius_mapping_and_matching_eye_fallback(radius):
    series = make_series(radius=radius)
    assert series.get_corneal_radius_metadata()['corneal_radius_mm'] == radius
    assert series.study.folders[7][0].data.c_curve_mm == radius
    for sl in series.slices.values():
        sl.folders.pop(39)
    info = series.get_corneal_radius_metadata()
    assert info['corneal_radius_mm'] == radius
    assert info['source'].startswith('Type7')


def test_default_volume_keeps_unknown_metric_scale_and_ordered_annotations():
    series = make_series()
    volume = series.get_volume()
    assert volume.shape == (3, 4, 8)
    assert volume.scale_unit == 'px' and volume.scale == (1, 1, 1)
    assert series.get_heyex_metadata()['OCT Image']['Scaling X'] is None
    assert series.get_heyex_metadata()['OCT Image']['Eye Length'] == 'Medium'
    layer = next(iter(volume.layers.values()))
    np.testing.assert_array_equal(layer.data[:, 0], [1, 2, 3])
    assert layer.manual == [False, True, False]
    assert len({float(volume.raw_data[i, 0, 0]) for i in range(3)}) == 3


def test_explicit_metric_calibration_uses_full_width_without_mutating_defaults():
    series = make_series(missing_layers=True)
    metric = series.get_volume(optical_model=EMPIRICAL_OPTICAL_MODEL)
    q = EMPIRICAL_OPTICAL_MODEL.mm_per_degree(7.7, .9)
    assert metric.scale_unit == metric.localizer.scale_unit == 'mm'
    assert metric.scale_x == pytest.approx(q * 30 / 8)
    assert metric.scale_y == pytest.approx(.00387)
    assert metric.scale_z == pytest.approx(q)
    assert metric.meta['bscan_meta'][0]['pos_unit'] == 'mm'
    assert metric.layers == {}
    assert metric.meta['e2e_metadata']['findings']['calibration']['status'] == 'approximate'
    default = series.get_volume()
    assert default.scale_unit == 'px'
    assert default.localizer.scale_unit == '°'
    assert default.meta['bscan_meta'][0]['pos_unit'] == '°'


def test_reference_keys_do_not_round_timestamp_or_use_current_focus():
    timestamp = 2**60 + 123
    baseline = make_series(own_time=timestamp, reference_time=timestamp)
    follow = make_series(focus=-1.09, own_time=timestamp+1, reference_time=timestamp)
    assert follow.get_reference_key() == baseline.get_progression_key()
    assert follow.get_progression_key() != baseline.get_progression_key()
    assert follow.get_calibration_metadata(EMPIRICAL_OPTICAL_MODEL)['reference_status'] == 'unresolved'
    with pytest.raises(ValueError, match='unresolved'):
        follow.get_volume(optical_model=EMPIRICAL_OPTICAL_MODEL)
    resolved = follow.get_calibration_metadata(EMPIRICAL_OPTICAL_MODEL, baseline)
    assert resolved['effective_inputs']['focus_diopters'] == pytest.approx(.9)
    assert resolved['current_inputs']['focus_diopters'] == pytest.approx(-1.09)
    wrong = make_series(own_time=timestamp+2)
    with pytest.raises(ValueError, match='does not match'):
        follow.get_calibration_metadata(EMPIRICAL_OPTICAL_MODEL, wrong)


def test_external_reference_lookup_and_ambiguity():
    base = make_series(own_time=100)
    follow = make_series(own_time=101, reference_time=100)
    reader = object.__new__(HeE2eReader)
    reader.file_hierarchy = SimpleNamespace(patients={1: SimpleNamespace(studies={2: SimpleNamespace(series={3: follow})})})
    assert reader.resolve_reference(follow) is None
    assert reader.resolve_reference(follow, [base]) is base
    duplicate = make_series(own_time=100)
    with pytest.raises(ValueError, match='Ambiguous'):
        reader.resolve_reference(follow, [base, duplicate])


def test_reference_metric_geometry_uses_reference_localizer():
    base = make_series(own_time=100, reference_time=100)
    original = base.get_reference_angles()
    matrix = np.array([[.98, .02, 3], [-.01, 1.01, -2]])
    angles = original @ matrix[:, :2].T + matrix[:, 2] * 30 / 16
    follow = make_series(focus=-1.09, own_time=101, reference_time=100,
                         affine=matrix.ravel(), angles=angles)
    base.folders[1073741824][0].data.data = np.full((16, 16), 9, dtype=np.uint8)
    follow.folders[1073741824][0].data.data = np.full((16, 16), 17, dtype=np.uint8)
    metric = follow.get_volume(optical_model=EMPIRICAL_OPTICAL_MODEL, reference_series=base)
    q = EMPIRICAL_OPTICAL_MODEL.mm_per_degree(7.7, .9)
    np.testing.assert_allclose(metric.meta['bscan_meta'][0]['start_pos'],
                               (original[0, 0] + 15) * q, atol=1e-6)
    assert metric.localizer.data[0, 0] == 9
    default = follow.get_volume()
    assert default.localizer.data[0, 0] == 17
    np.testing.assert_array_equal(metric.raw_data, default.raw_data)


def test_centered_registration_with_pixel_translation_roundtrips():
    original = np.array([[[-15, 1], [15, 1]], [[-15, -1], [15, -1]]], dtype=float)
    affine = np.array([[.98, .02, 3], [-.01, 1.01, -2]])
    registered = original @ affine[:, :2].T + affine[:, 2] * 30 / 1536
    actual = inverse_localizer_registration(registered, affine, 30, 1536)
    np.testing.assert_allclose(actual, original, atol=1e-12)
    with pytest.raises(ValueError, match='Singular'):
        inverse_localizer_registration(original, np.zeros((2, 3)), 30, 1536)


def test_unsupported_registration_is_not_silently_calibrated():
    series = make_series(own_time=100, reference_time=100)
    folder = series.folders[10025][0]
    raw = bytearray(folder.reader.file_object.getvalue())
    struct.pack_into('<I', raw, folder.start + 60, 99)
    folder.reader.file_object = BytesIO(raw)
    assert series.get_registration_metadata()['status'] == 'unsupported'
    assert series.get_calibration_metadata(EMPIRICAL_OPTICAL_MODEL)['reference_status'] == 'unsupported_registration'
    with pytest.raises(ValueError, match='unsupported_registration'):
        series.get_volume(optical_model=EMPIRICAL_OPTICAL_MODEL)


def test_nonuniform_sampling_is_not_collapsed_to_one_scale():
    series = make_series()
    series._bscanmeta_items()[1].size_x = 9
    with pytest.raises(ValueError, match='Nonuniform lateral'):
        series.get_calibration_metadata(EMPIRICAL_OPTICAL_MODEL)


def test_unknown_preset_and_invalid_optical_model():
    assert type10004_format.parse(bscan_payload(preset=123)).eye_length_preset is None
    with pytest.raises(ValueError):
        E2EOpticalModel('invalid', 0, -1, 0).mm_per_degree(7.7, 0)
    with pytest.raises(ValueError):
        RESEARCH_OPTICAL_MODEL.mm_per_degree(float('nan'), 0)


def test_empty_series_calibration_inputs_are_unavailable():
    empty = E2ESeriesStructure(10)
    empty.study = make_series().study
    metadata = empty.get_calibration_metadata()
    assert metadata['current_inputs']['focus_raw'] is None
    assert metadata['current_inputs']['corneal_radius_mm'] is None


def test_unresolved_reference_fails_before_loading_oct_pixels(monkeypatch):
    follow = make_series(own_time=101, reference_time=100)
    monkeypatch.setattr(follow, 'get_bscans', lambda: pytest.fail('OCT pixels must not be loaded'))
    with pytest.raises(ValueError, match='unresolved'):
        follow.get_volume(optical_model=EMPIRICAL_OPTICAL_MODEL)


def test_group_membership_is_not_a_reference_order_claim():
    raw = 'Default\0'.encode('utf-16-le') + struct.pack('<H', 2)
    raw += struct.pack('<4iH', 1, 2, 3, -1, 65535) + struct.pack('<4iH', 1, 2, 4, -1, 65535)
    group = datacontainer_format.parse(container(6, raw)).item.membership()
    assert [m['series_id'] for m in group['members']] == [3, 4]
    assert 'unverified' in group['status']


def test_auxiliary_image_and_opaque_opening_record():
    raw = struct.pack('<5I', 26, 33620481, 6, 2, 3) + bytes(range(6))
    image = datacontainer_format.parse(container(1073751824, raw)).item
    assert image.data.shape == (2, 3)
    opaque = datacontainer_format.parse(container(10020, bytes(20))).item
    assert opaque.raw == bytes(20)
