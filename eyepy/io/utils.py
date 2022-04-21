# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta, timezone
from typing import MutableMapping, Tuple, Union

import numpy as np
from skimage import transform

from eyepy.core.eyemeta import EyeBscanMeta, EyeEnfaceMeta, EyeVolumeMeta
from eyepy.io.lazy import LazyVolume

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
    date = elements[0]
    year = int(date.find("Year").text)
    month = int(date.find("Month").text)
    day = int(date.find("Day").text)
    return datetime(year, month, day).date()


def _compute_localizer_oct_transform(
    volume_meta: MutableMapping,
    enface_meta: MutableMapping,
    volume_shape: Tuple[int, int, int],
):
    bscan_meta = volume_meta["bscan_meta"]
    size_z, size_y, size_x = volume_shape
    # Points in oct space as row/column indices
    src = np.array(
        [
            [0, 0],  # Top left
            [0, size_x - 1],  # Top right
            [size_z - 1, 0],  # Bottom left
            [size_z - 1, size_x - 1],  # Bottom right
        ]
    )

    # Respective points in enface space as x/y coordinates
    scale = np.array([enface_meta["scale_x"], enface_meta["scale_y"]])
    dst = np.array(
        [
            bscan_meta[-1]["start_pos"] / scale,  # Top left
            bscan_meta[-1]["end_pos"] / scale,  # Top right
            bscan_meta[0]["start_pos"] / scale,  # Bottom left
            bscan_meta[0]["end_pos"] / scale,  # Bottom right
        ]
    )

    # Switch from row/column indices to x/y coordinates
    src = src[:, [1, 0]]
    return transform.estimate_transform("affine", src, dst)


def _get_enface_meta(lazy_volume: LazyVolume):
    return EyeEnfaceMeta(
        scale_x=lazy_volume.meta["ScaleXSlo"],
        scale_y=lazy_volume.meta["ScaleYSlo"],
        scale_unit="mm",
        modality="NIR",
        laterality=lazy_volume.meta["ScanPosition"],
        field_size=lazy_volume.meta["FieldSizeSlo"],
        scan_focus=lazy_volume.meta["ScanFocus"],
        visit_date=lazy_volume.meta["VisitDate"],
        exam_time=lazy_volume.meta["ExamTime"],
    )


def _get_volume_meta(lazy_volume: LazyVolume):
    bscan_meta = [
        EyeBscanMeta(
            quality=b.meta["Quality"],
            start_pos=(b.meta["StartX"], b.meta["StartY"]),
            end_pos=(b.meta["EndX"], b.meta["EndY"]),
            pos_unit="mm",
        )
        for b in lazy_volume
    ]

    if not lazy_volume.ScanPattern == 1:
        # Check if all B-scans are parallel and have the same distance. They might be rotated though
        dist_func = lambda a, b: np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        start_distances = [
            dist_func(bscan_meta[i]["start_pos"], bscan_meta[i + 1]["start_pos"])
            for i in range(len(bscan_meta) - 1)
        ]
        end_distances = [
            dist_func(bscan_meta[i]["end_pos"], bscan_meta[i + 1]["end_pos"])
            for i in range(len(bscan_meta) - 1)
        ]
        if not len(start_distances) == len(end_distances) == 1:
            msg = "B-scans are not equally spaced. Projections into the enface space are distorted."
            logger.warning(msg)
        bscan_distance = start_distances[0]
    else:
        bscan_distance = 0

    return EyeVolumeMeta(
        scale_x=lazy_volume.meta["ScaleX"],
        scale_y=lazy_volume.meta["ScaleY"],
        scale_z=bscan_distance,
        scale_unit="mm",
        laterality=lazy_volume.meta["ScanPosition"],
        visit_date=lazy_volume.meta["VisitDate"],
        exam_time=lazy_volume.meta["ExamTime"],
        bscan_meta=bscan_meta,
    )
