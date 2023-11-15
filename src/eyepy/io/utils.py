from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import logging
import sys
from typing import Optional, Union

import construct as cs
import numpy as np
from skimage import transform
from skimage.transform._geometric import _GeometricTransform

from eyepy.core.eyemeta import EyeBscanMeta

logger = logging.getLogger(__name__)


def _get_meta_attr(meta_attr):

    def prop_func(self):
        return getattr(self.meta, meta_attr)

    return property(prop_func)


def _clean_ascii(unpacked: tuple):
    return unpacked[0].decode('ascii').rstrip('\x00')


def _get_first(unpacked: tuple):
    return unpacked[0]


def _date_in_seconds(
    dt: datetime,
    epoche: datetime = datetime.utcfromtimestamp(0),
    second_frac: Union[float, int] = 1,
):
    seconds = (dt - epoche).total_seconds() / second_frac
    if not seconds.is_integer():
        raise ValueError('The resulting number needs to be a whole number')
    return int(seconds)


def _date_from_seconds(
    s: int,
    epoche: datetime = datetime.utcfromtimestamp(0),
    second_frac: Union[float, int] = 1,
):
    return epoche + timedelta(seconds=s * second_frac)


def _get_first_as_int(elements):
    if elements:
        return int(elements[0].text)
    else:
        return None


def _get_first_as_float(elements):
    if elements:
        return float(elements[0].text)
    else:
        return None


def _get_first_as_str(elements):
    if elements:
        return elements[0].text
    else:
        return None


def _get_datetime_from_xml(elements):
    date = elements[0].find('Series/ExamDate/Date')
    time = elements[0].find('Image/AcquisitionTime/Time')
    if date is None or time is None:
        return None

    year = int(date.find('Year').text)
    month = int(date.find('Month').text)
    day = int(date.find('Day').text)
    hour = int(time.find('Hour').text)
    minute = int(time.find('Minute').text)
    second = int(float(time.find('Second').text))
    utc_bias = int(time.find('UTCBias').text)
    tz = timezone(timedelta(minutes=utc_bias))
    return datetime(year, month, day, hour, minute, second, tzinfo=tz)


def _get_date_from_xml(elements):
    try:
        date = elements[0]
        year = int(date.find('Year').text)
        month = int(date.find('Month').text)
        day = int(date.find('Day').text)
    except IndexError:
        year = month = day = 1

    return datetime(year, month, day).isoformat()


def _compute_localizer_oct_transform(
    volume_meta: MutableMapping,
    enface_meta: MutableMapping,
    volume_shape: tuple[int, int, int],
) -> GeometricTransform:
    bscan_meta = volume_meta['bscan_meta']
    size_z, size_y, size_x = volume_shape
    # Points in oct space as row/column indices
    src = np.array([
        [0, 0],  # Top left
        [0, size_x - 1],  # Top right
        [size_z - 1, 0],  # Bottom left
        [size_z - 1, size_x - 1],  # Bottom right
    ])

    # Respective points in enface space as x/y coordinates
    scale = np.array([enface_meta['scale_x'], enface_meta['scale_y']])
    dst = np.array([
        bscan_meta[-1]['start_pos'] / scale,  # Top left
        bscan_meta[-1]['end_pos'] / scale,  # Top right
        bscan_meta[0]['start_pos'] / scale,  # Bottom left
        bscan_meta[0]['end_pos'] / scale,  # Bottom right
    ])

    # Switch from row/column indices to x/y coordinates by flipping last axis of src
    src = np.flip(src, axis=1)
    # src = src[:, [1, 0]]
    return transform.estimate_transform('affine', src, dst)


def get_date_adapter(construct, epoch, second_frac):

    class DateAdapter(cs.Adapter):

        def _decode(self, obj, context, path):
            return (epoch + timedelta(seconds=obj * second_frac)).isoformat()

        def _encode(self, obj, context, path):
            return (datetime.fromisoformat(obj) -
                    epoch).total_seconds() / second_frac

    return DateAdapter(construct)


class BscanAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype='float32',
                          shape=(context._.size_y, context._.size_x))

    def _encode(self, obj, context, path):
        return obj.tobytes()


class LocalizerAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype='uint8',
                          shape=(context.size_y_slo, context.size_x_slo))

    def _encode(self, obj, context, path):
        return obj.tobytes()


class SegmentationsAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype='float32',
                          shape=(context.num_seg, context._.size_x))

    def _encode(self, obj, context, path):
        return obj.tobytes()


IntDate = get_date_adapter(cs.Int64ul, datetime(1601, 1, 1), 1e-7)
FloatDate = get_date_adapter(cs.Float64l, datetime(1899, 12, 30), 60 * 60 * 24)
Localizer = LocalizerAdapter(cs.Bytes(cs.this.size_x_slo * cs.this.size_y_slo))
Segmentations = SegmentationsAdapter(
    cs.Bytes(cs.this.num_seg * cs.this._.size_x * 4))
Bscan = BscanAdapter(cs.Bytes(cs.this._.size_y * cs.this._.size_x * 4))


def get_bscan_spacing(bscan_meta: list[EyeBscanMeta]):
    # Check if all B-scans are parallel and have the same distance. They might be rotated though
    dist_func = lambda a, b: np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    start_distances = [
        dist_func(bscan_meta[i]['start_pos'], bscan_meta[i + 1]['start_pos'])
        for i in range(len(bscan_meta) - 1)
    ]
    end_distances = [
        dist_func(bscan_meta[i]['end_pos'], bscan_meta[i + 1]['end_pos'])
        for i in range(len(bscan_meta) - 1)
    ]
    if not np.allclose(start_distances[0],
                       np.array(start_distances + end_distances),
                       rtol=4e-2):
        msg = 'B-scans are not equally spaced. Projections into the enface space are distorted.'
        logger.warning(msg)
    return np.mean(start_distances + end_distances)


def find_int(bytestring: bytes,
             value: int,
             signed: Optional[Union[bool, str, list[str]]] = None,
             endian: Optional[str] = None,
             bits: Optional[Union[int, list[int], str, list[str]]] = None,
             rtol: float = 1e-05,
             atol: float = 1e-08) -> dict[str, list[int]]:
    """Find all occurrences of an integer in a byte string.

    Args:
        bytestring: The byte string to search.
        value: The integer to search for.
        signed: Whether the integer is signed or not. If not specified, the
            integer is assumed to be signed if it is negative, otherwise signed and unsigned are searched for.
        endian: "l" for little and "b" for big, the endianness of the integer.
            If not specified, the endianness is assumed to be the same as the endianness of the system.
        bits: The number of bits in the integer. If not specified, 8, 16, 24, 32, and 64
            bit integers are searched for.
        rtol: The relative tolerance parameter for matching a value (see numpy.isclose).
        atol: The absolute tolerance parameter for matching a value (see numpy.isclose).

    Returns:
        A dictionary where the key is the type and the value, a list of offsets for which the searched value was found
    """

    # construct format strings
    if signed is None:
        signed = ['s'] if value < 0 else ['s', 'u']
    elif isinstance(signed, bool):
        signed = ['s'] if signed else ['u']
    elif isinstance(signed, str):
        signed = [signed]
    if endian is None:
        endian = sys.byteorder[0]  # first letter of endianness
    if bits is None:
        bits = ['8', '16', '24', '32', '64']
    elif isinstance(bits, int):
        bits = [str(bits)]
    elif isinstance(bits, str):
        bits = [bits]
    elif isinstance(bits, list):
        bits = [str(b) for b in bits]

    # Build a list of all format strings
    formats = [(int(b), f'Int{b}{s}{endian}') for b in bits for s in signed]

    # find all occurrences
    results = defaultdict(list)
    for bts, fmt_string in formats:
        format = getattr(cs, fmt_string)

        # Parse the bytestring with multiple byte offsets depending on the format
        for offset in range(bts // 8):

            # Calculate the number of items that can be parsed
            count = (len(bytestring) - offset) // (bts // 8)
            if count <= 0:
                continue
            data = np.array(cs.Array(count, format).parse(bytestring[offset:]))

            # Find all occurrences
            hits = np.nonzero(np.isclose(data, value, rtol=rtol, atol=atol))
            res = [(pos * (bts // 8)) + offset + 1 for pos in hits[0]]
            if res:
                results[fmt_string] += res

    results = {**results}
    return results


def find_float(bytestring: bytes,
               value: float,
               endian: Optional[str] = None,
               bits: Optional[Union[int, list[int], str, list[str]]] = None,
               rtol: float = 1e-05,
               atol: float = 1e-08) -> dict[str, list[int]]:
    """Find all occurrences of a float in a byte string.

    Args:
        bytestring: The byte string to search.
        value: The float to search for.
        endian: "l" for little and "b" for big, the endianness of the float.
            If not specified, the endianness is assumed to be the same as the endianness of the system.
        bits: The number of bits in the float. If not specified, 16, 32 and 64
            bit floats are searched for.
        rtol: The relative tolerance parameter for matching a value (see numpy.isclose).
        atol: The absolute tolerance parameter for matching a value (see numpy.isclose).

    Returns:
        A dictionary where the key is the type and the value, a list of offsets for which the searched value was found
    """

    # construct format strings
    if endian is None:
        endian = sys.byteorder[0]  # first letter of endianness
    if bits is None:
        bits = ['16', '32', '64']
    elif isinstance(bits, int):
        bits = [str(bits)]
    elif isinstance(bits, str):
        bits = [bits]
    elif isinstance(bits, list):
        bits = [str(b) for b in bits]

    # Build a list of all format strings
    formats = [(int(b), f'Float{b}{endian}') for b in bits]

    # find all occurrences
    results = defaultdict(list)
    for bts, fmt_string in formats:
        format = getattr(cs, fmt_string)

        # Parse the bytestring with multiple byte offsets depending on the format
        for offset in range(bts // 8):
            # Calculate the number of items that can be parsed
            count = (len(bytestring) - offset) // (bts // 8)
            if count <= 0:
                continue
            data = np.array(cs.Array(count, format).parse(bytestring[offset:]))

            # Find all occurrences for this format
            hits = np.nonzero(np.isclose(data, value, rtol, atol))
            res = [(pos * (bts // 8)) + offset + 1 for pos in hits[0]]
            if res:
                results[fmt_string] += res

    results = {**results}
    return results
