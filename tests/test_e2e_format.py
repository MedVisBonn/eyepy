import struct

import numpy as np
import pytest

pytest.importorskip('construct_typed')

from eyepy.io.he.e2e_format import chunk_format
from eyepy.io.he.e2e_format import datacontainer_format
from eyepy.io.he.e2e_format import type7_format
from eyepy.io.he.e2e_format import Type39
from eyepy.io.he.e2e_format import type10004_format


def test_e2e_structures_import_with_supported_construct_typing():
    """Importing all E2E formats catches construct-typing API
    incompatibilities."""
    assert chunk_format is not None


def test_chunk_seek_field_parses_without_becoming_an_init_argument():
    """Seek-only fields remain parseable with construct-typing 0.8."""
    header = struct.pack(
        '<12sI10HIIII', b'CHUNK', 1, *([0] * 10), 1, 0, 0, 0
    )
    folder = struct.pack(
        '<IIIIiiiiHHII', 0, 96, 0, 0, 1, 2, 3, 4, 1, 0, 9, 0
    )

    parsed = chunk_format.parse(header + folder + bytes(60))

    assert len(parsed.folders) == 1
    assert parsed.folders[0].type.name == 'patient'


@pytest.mark.parametrize('focus', [-1.51, -1.09, -0.49, -0.06, 0, 0.36, 0.82, 0.90])
def test_focus_uses_full_float64_without_shifting_quality(focus):
    raw = bytearray(438)
    struct.pack_into('<d', raw, 136, focus / 0.001043)
    struct.pack_into('<d', raw, 144, 2500)
    struct.pack_into('<f', raw, 156, 32)

    meta = type10004_format.parse(raw)

    assert meta.focus_diopters == pytest.approx(focus)
    assert meta.eye_length_preset_raw == 2500
    assert meta.eye_length_preset == 'Long'
    assert meta.quality_ui == 32
    assert len(meta.rest) == 278
    assert type10004_format.build(meta) == raw


@pytest.mark.parametrize('radius', [6.2, 7.7, 9.0])
def test_corneal_radius_offsets(radius):
    study_raw = bytearray(68)
    struct.pack_into('<d', study_raw, 1, 123)
    struct.pack_into('<d', study_raw, 17, radius)
    slice_raw = bytearray(600)
    struct.pack_into('<d', slice_raw, 457, radius)

    study = type7_format.parse(study_raw)

    assert study.corneal_radius_mm == radius
    assert Type39(raw=bytes(slice_raw)).corneal_radius_mm == radius
    assert type7_format.build(study) == study_raw


def test_short_localizer_settings_have_no_corneal_radius():
    assert Type39(raw=bytes(464)).corneal_radius_mm is None


@pytest.mark.parametrize('eye,values,lens,pupil', [
    ('R', (5, 1, 6.2, 6, 7, 2, 3), 1, 4),
    ('R', (3.4, 7.3, 6.2, 2.9, 1.1, 6.9, 5.2), 2, 4.6),
    ('R', (3.44, 7.38, 6.2, 2.93, 1.14, 6.97, 5.26), 3, 4.65),
    ('L', (0, 0, 6.2, 0, 0, 0, 0), 0, 0),
])
def test_type7_fields_match_controlled_eye_data(eye, values, lens, pupil):
    # Non-identifying values from the controlled Eye Data screenshots.
    # Seven doubles precede the lens code; the pupil double follows it.
    raw = struct.pack('<B7dHdB', ord(eye), *values, lens, pupil, 1)

    study = type7_format.parse(raw)

    assert study.eye_side.name == ('OD' if eye == 'R' else 'OS')
    assert study.iop_mmHg == values[0]
    assert study.refraction_dpt == values[1]
    assert study.corneal_radius_mm == values[2]
    assert study.vfield_mean == values[3]
    assert study.vfield_var == values[4]
    assert study.cylinder_dpt == values[5]
    assert study.axis_deg == values[6]
    assert study.corrective_lens == lens
    assert study.pupil_size_mm == pupil
    assert study.rest == bytes([1])
    assert type7_format.build(study) == raw


def _type9_container(firstname, surname, patient_id):
    item = (
        firstname.ljust(31, b'\x00')[:31] +
        surname.ljust(66, b'\x00')[:66] +
        struct.pack('<I', 0) +
        b'M' +
        patient_id.ljust(25, b'\x00')[:25]
    )
    item_size = len(item)
    header = struct.pack(
        '<12sIIIIIiiiiHHII',
        b'CONTAINER',
        0,
        0,
        0,
        item_size,
        0,
        1,
        2,
        3,
        4,
        1,
        0,
        9,
        0,
    )
    return header + item


def _type10019_container(values, padding=()):
    item_size = 16 + len(padding) * 4 + len(values) * 4
    header = struct.pack(
        '<12sIIIIIiiiiHHII',
        b'CONTAINER',
        0,
        0,
        0,
        item_size,
        0,
        1,
        2,
        3,
        4,
        1,
        0,
        10019,
        0,
    )
    item = struct.pack('<IIII', 2, 7, 5, len(values))
    item += struct.pack(f'<{len(padding)}I', *padding) if padding else b''
    item += struct.pack(f'<{len(values)}f', *values)
    return header + item


@pytest.mark.parametrize('padding', [(0, ), (0, 0, 0, 0, 0)])
def test_type10019_parses_layer_data_after_variable_padding(padding):
    values = (1.25, 2.5, 3.75)

    parsed = datacontainer_format.parse(_type10019_container(values, padding))

    assert parsed.item.width == len(values)
    assert parsed.item.manual_edit_raw == padding[0]
    assert list(parsed.item.unknown2) == list(padding[1:])
    np.testing.assert_allclose(parsed.item.data, values)


def test_type9_parses_latin1_patient_strings():
    parsed = datacontainer_format.parse(
        _type9_container(b'J\xf6rg', b'M\xfcller', b'P\xc4T-1'))

    assert parsed.item.firstname == 'J\xf6rg'
    assert parsed.item.surname == 'M\xfcller'
    assert parsed.item.patient_id == 'P\xc4T-1'
