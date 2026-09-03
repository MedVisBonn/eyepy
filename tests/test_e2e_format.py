import struct

import numpy as np
import pytest

pytest.importorskip('construct_typed')

from eyepy.io.he.e2e_format import chunk_format
from eyepy.io.he.e2e_format import datacontainer_format


def test_e2e_structures_import_with_supported_construct_typing():
    """Importing all E2E formats catches construct-typing API
    incompatibilities."""
    assert chunk_format is not None


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


@pytest.mark.parametrize('padding', [(), (0, 0, 0, 0, 0)])
def test_type10019_parses_layer_data_after_variable_padding(padding):
    values = (1.25, 2.5, 3.75)

    parsed = datacontainer_format.parse(_type10019_container(values, padding))

    assert parsed.item.width == len(values)
    assert list(parsed.item.unknown2) == list(padding)
    np.testing.assert_allclose(parsed.item.data, values)


def test_type9_parses_latin1_patient_strings():
    parsed = datacontainer_format.parse(
        _type9_container(b'J\xf6rg', b'M\xfcller', b'P\xc4T-1'))

    assert parsed.item.firstname == 'J\xf6rg'
    assert parsed.item.surname == 'M\xfcller'
    assert parsed.item.patient_id == 'P\xc4T-1'
