from datetime import datetime, timedelta, timezone
from struct import unpack, calcsize
from typing import Union


def _create_properties(version, version_dict):
    """Dynamically create properties for different version of the .vol export"""
    fields = [entry[0] for entry in version_dict[version]]
    fmts = [entry[1] for entry in version_dict[version]]
    funcs = [entry[2] for entry in version_dict[version]]

    attribute_dict = {}
    for field, func in zip(fields, funcs):
        attribute_dict[field] = _create_property(field, func)

    attribute_dict["_fmt"] = fmts
    attribute_dict["_meta_fields"] = fields
    attribute_dict["_Version"] = version

    return attribute_dict


def _create_properties_xml(version, version_dict):
    """Dynamically create properties for different version of the .xml export"""
    fields = [entry[0] for entry in version_dict[version]]
    locs = [entry[1] for entry in version_dict[version]]
    funcs = [entry[2] for entry in version_dict[version]]

    attribute_dict = {}
    for field, func in zip(fields, funcs):
        attribute_dict[field] = _create_property_xml(field, func)

    attribute_dict["_locs"] = locs
    attribute_dict["_meta_fields"] = fields
    attribute_dict["_Version"] = version

    return attribute_dict


def _create_property_xml(field, func):
    def prop_func(self):
        field_position = self._meta_fields.index(field)
        loc = self._locs[field_position]
        if getattr(self, f"_{field}") is None:
            attr = func(self._root[0].findall(loc))
            setattr(self, f"_{field}", attr)

        return getattr(self, f"_{field}")

    return property(prop_func)


def _create_property(field, func):
    def prop_func(self):
        field_position = self._meta_fields.index(field)
        startpos = self._startpos + calcsize(
            "=" + "".join(self._fmt[:field_position]))
        fmt = self._fmt[field_position]

        if getattr(self, f"_{field}") is None:
            self._file_obj.seek(startpos, 0)
            content = self._file_obj.read(calcsize(fmt))
            attr = func(unpack(fmt, content))

            setattr(self, f"_{field}", attr)

        return getattr(self, f"_{field}")

    return property(prop_func)


def _get_meta_attr(meta_attr):
    def prop_func(self):
        return getattr(self.meta, meta_attr)

    return property(prop_func)


def _clean_ascii(unpacked: tuple):
    return unpacked[0].decode("ascii").rstrip("\x00")


def _get_first(unpacked: tuple):
    return unpacked[0]


def _date_in_seconds(dt: datetime,
                     epoche: datetime = datetime.utcfromtimestamp(0),
                     second_frac: Union[float, int] = 1):
    seconds = (dt - epoche).total_seconds() / second_frac
    if not seconds.is_integer():
        raise ValueError("The resulting number needs to be a whole number")
    return int(seconds)


def _date_from_seconds(s: int, epoche: datetime = datetime.utcfromtimestamp(0),
                       second_frac: Union[float, int] = 1):
    return epoche + timedelta(seconds=s * second_frac)


def _get_first_as_int(elements):
    return int(elements[0].text)


def _get_first_as_float(elements):
    return float(elements[0].text)


def _get_first_as_str(elements):
    return elements[0].text


def _get_datetime_from_xml(elements):
    date = elements[0].find("ReferenceSeries/ExamDate/Date")
    time = elements[0].find("Image/AcquisitionTime/Time")

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
