# -*- coding: utf-8 -*-
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import logging
from typing import List, MutableMapping, Tuple, Union

import construct as cs
import numpy as np
from skimage import transform
from skimage.transform._geometric import GeometricTransform

from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeVolumeMeta

logger = logging.getLogger(__name__)


def _get_meta_attr(meta_attr):

    def prop_func(self):
        return getattr(self.meta, meta_attr)

    return property(prop_func)


def _clean_ascii(unpacked: tuple):
    return unpacked[0].decode("ascii").rstrip("\x00")


def _get_first(unpacked: tuple):
    return unpacked[0]


def _date_in_seconds(
    dt: datetime,
    epoche: datetime = datetime.utcfromtimestamp(0),
    second_frac: Union[float, int] = 1,
):
    seconds = (dt - epoche).total_seconds() / second_frac
    if not seconds.is_integer():
        raise ValueError("The resulting number needs to be a whole number")
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
    date = elements[0].find("Series/ExamDate/Date")
    time = elements[0].find("Image/AcquisitionTime/Time")
    if date is None or time is None:
        return None

    year = int(date.find("Year").text)
    month = int(date.find("Month").text)
    day = int(date.find("Day").text)
    hour = int(time.find("Hour").text)
    minute = int(time.find("Minute").text)
    second = int(float(time.find("Second").text))
    utc_bias = int(time.find("UTCBias").text)
    tz = timezone(timedelta(minutes=utc_bias))
    return datetime(year, month, day, hour, minute, second, tzinfo=tz)


def _get_date_from_xml(elements):
    try:
        date = elements[0]
        year = int(date.find("Year").text)
        month = int(date.find("Month").text)
        day = int(date.find("Day").text)
    except IndexError:
        year = month = day = 1

    return datetime(year, month, day).date()


def _compute_localizer_oct_transform(
    volume_meta: MutableMapping,
    enface_meta: MutableMapping,
    volume_shape: Tuple[int, int, int],
) -> GeometricTransform:
    bscan_meta = volume_meta["bscan_meta"]
    size_z, size_y, size_x = volume_shape
    # Points in oct space as row/column indices
    src = np.array([
        [0, 0],  # Top left
        [0, size_x - 1],  # Top right
        [size_z - 1, 0],  # Bottom left
        [size_z - 1, size_x - 1],  # Bottom right
    ])

    # Respective points in enface space as x/y coordinates
    scale = np.array([enface_meta["scale_x"], enface_meta["scale_y"]])
    dst = np.array([
        bscan_meta[-1]["start_pos"] / scale,  # Top left
        bscan_meta[-1]["end_pos"] / scale,  # Top right
        bscan_meta[0]["start_pos"] / scale,  # Bottom left
        bscan_meta[0]["end_pos"] / scale,  # Bottom right
    ])

    # Switch from row/column indices to x/y coordinates by flipping last axis of src
    src = np.flip(src, axis=1)
    # src = src[:, [1, 0]]
    return transform.estimate_transform("affine", src, dst)


def get_date_adapter(construct, epoch, second_frac):

    class DateAdapter(cs.Adapter):

        def _decode(self, obj, context, path):
            return epoch + timedelta(seconds=obj * second_frac)

        def _encode(self, obj, context, path):
            return int((obj - epoch).total_seconds() / second_frac)

    return DateAdapter(construct)


class BscanAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype="float32",
                          shape=(context._.size_y, context._.size_x))

    def _encode(self, obj, context, path):
        return obj.tobytes()


class LocalizerAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype="uint8",
                          shape=(context.size_y_slo, context.size_x_slo))

    def _encode(self, obj, context, path):
        return obj.tobytes()


class SegmentationsAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype="float32",
                          shape=(17, context._.size_x))

    def _encode(self, obj, context, path):
        return obj.tobytes()


IntDate = get_date_adapter(cs.Int64ul, datetime(1601, 1, 1), 1e-7)
FloatDate = get_date_adapter(cs.Float64l, datetime(1899, 12, 30), 60 * 60 * 24)
Localizer = LocalizerAdapter(cs.Bytes(cs.this.size_x_slo * cs.this.size_y_slo))
Segmentations = SegmentationsAdapter(cs.Bytes(17 * cs.this._.size_x * 4))
Bscan = BscanAdapter(cs.Bytes(cs.this._.size_y * cs.this._.size_x * 4))


def get_bscan_spacing(bscan_meta: List[EyeBscanMeta]):
    # Check if all B-scans are parallel and have the same distance. They might be rotated though
    dist_func = lambda a, b: np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    start_distances = [
        dist_func(bscan_meta[i]["start_pos"], bscan_meta[i + 1]["start_pos"])
        for i in range(len(bscan_meta) - 1)
    ]
    end_distances = [
        dist_func(bscan_meta[i]["end_pos"], bscan_meta[i + 1]["end_pos"])
        for i in range(len(bscan_meta) - 1)
    ]
    if not np.allclose(start_distances[0],
                       np.array(start_distances + end_distances)):
        msg = "B-scans are not equally spaced. Projections into the enface space are distorted."
        logger.warning(msg)
    return np.mean(start_distances + end_distances)
