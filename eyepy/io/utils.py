from datetime import datetime, timedelta
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
