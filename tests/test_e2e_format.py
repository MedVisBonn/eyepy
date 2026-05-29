import struct

import numpy as np
import pytest

pytest.importorskip('construct_typed')

from eyepy.io.he.e2e_format import datacontainer_format


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
