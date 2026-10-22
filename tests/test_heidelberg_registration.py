import struct

import numpy as np
import pytest

from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.io.he.e2e_format import Bscan
from eyepy.io.he.e2e_format import type10010_format
from eyepy.io.he.e2e_format import type10012_format


def matrix(tx=0, ty=0, shear=0):
    return [[1, 0, tx], [shear, 1, ty]]


def registration_payload(values):
    raw = bytearray(100)
    for offset, value in zip([4, 8, 12, 28, 32, 36],
                             np.asarray(values).ravel()):
        struct.pack_into('<f', raw, offset, value)
    return bytes(raw)


def test_unsigned_float_mantissa_order_zero_and_invalid_code():
    words = np.array([0, 1023, 1024, 0xf800, 0xf801, 0xfbff, 0xfc00, 0xffff],
                     '<u2')
    decoded = Bscan.parse(words.tobytes(),
                          height=1,
                          width=len(words),
                          n_values=len(words))
    expected = [0, 0, 2**-62, .5, .50048828125, .99951171875, 1, 1.9990234375]
    np.testing.assert_array_equal(decoded[0], np.array(expected, np.float32))
    encoded = Bscan.build(decoded[:, [0, 4, 6, 7]], n_values=4)
    np.testing.assert_array_equal(np.frombuffer(encoded, '<u2'),
                                  [0, 0xf801, 0xfc00, 0xffff])


def test_normalization_and_registration_structures_round_trip():
    raw = bytes(12) + struct.pack('<I3f', 3, 2, 4, 8) + bytes(16)
    parsed = type10010_format.parse(raw)
    assert parsed.normalization == [2, 4, 8]
    assert type10010_format.build(parsed) == raw
    raw = registration_payload(matrix(tx=.25, ty=-1, shear=.125))
    parsed = type10012_format.parse(raw)
    assert parsed.matrix == matrix(tx=.25, ty=-1, shear=.125)
    assert type10012_format.build(parsed) == raw


def test_half_pixel_boundary_clamps_lower_edge_and_excludes_upper_edge():
    transform = HeidelbergBscanTransform([1], [matrix(tx=.5, ty=-.5)])
    data = np.array([[.125, .25, .375], [.5, .625, .75]], np.float32)
    registered = transform(data, 0)
    assert registered[0, 0] == .1875
    assert registered[0, -1] == np.finfo(np.float32).max
    assert registered[1, 0] == .375


@pytest.mark.parametrize('output,corner,values,expected', [
    ((307, 991), (279, 991), [[1.6961712390184402e-7, 0],
                              [5.714595317840576e-6, 1.1339783668518066e-5]
                             ], 8.046627044677734e-5),
    ((318, 67), (265, 67), [[3.8817524909973145e-6, 1.8149614334106445e-5],
                            [2.7045607566833496e-5, 1.9416213035583496e-5]
                           ], .0002313852310180664),
])
def test_observed_rounding_cases(output, corner, values, expected):
    # Sparse, non-identifying numerical fixtures. Expected values come from
    # VOL, independently of this implementation. These catch intermediate rounding.
    source = np.zeros((496, 1536), np.float32)
    y, x = corner
    source[y:y + 2, x:x + 2] = values
    transform = HeidelbergBscanTransform(
        [12.229011469742101],
        [matrix(.3904285430908203, -33.24694061279297, .027252664789557457)])
    assert transform(source, 0)[output] == np.float32(expected)
