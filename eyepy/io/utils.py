# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta, timezone
from typing import Union

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
