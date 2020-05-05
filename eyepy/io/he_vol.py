import functools
import os
from datetime import datetime, timedelta
from struct import unpack, calcsize

import numpy as np

"""
Inspired by:
https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""


@functools.lru_cache(maxsize=4, typed=False)
def get_slo(filepath, oct_meta=None):
    return Slo(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_bscans(filepath, oct_meta=None):
    return Bscans(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_octmeta(filepath):
    return OctMeta(filepath)


def _clean_ascii(unpacked: tuple):
    return unpacked[0].decode("ascii").rstrip("\x00")


def _get_first(unpacked: tuple):
    return unpacked[0]


def _parse_date(unpacked: tuple):
    return datetime.fromtimestamp((unpacked[0] - 25569) * 24 * 60 * 60)


OCT_HDR_VERSIONS = {
    "HSF-OCT-103": [("Version", "12s", _clean_ascii),
                    ("SizeX", "I", _get_first),
                    ("NumBScans", "I", _get_first),
                    ("SizeZ", "I", _get_first),
                    ("ScaleX", "d", _get_first),
                    ("Distance", "d", _get_first),
                    ("ScaleZ", "d", _get_first),
                    ("SizeXSlo", "I", _get_first),
                    ("SizeYSlo", "I", _get_first),
                    ("ScaleXSlo", "d", _get_first),
                    ("ScaleYSlo", "d", _get_first),
                    ("FieldSizeSlo", "I", _get_first),
                    ("ScanFocus", "d", _get_first),
                    ("ScanPosition", "4s", _clean_ascii),
                    ("ExamTime", "Q", lambda x: (
                            datetime(1601, 1, 1) + timedelta(
                            seconds=x[0] * 1e-7))),
                    ("ScanPattern", "i", _get_first),
                    ("BScanHdrSize", "I", _get_first),
                    ("ID", "16s", _clean_ascii),
                    ("ReferenceID", "16s", _clean_ascii),
                    ("PID", "I", _get_first),
                    ("PatientID", "24s", _clean_ascii),
                    ("DOB", "d", _parse_date),
                    ("VID", "I", _get_first),
                    ("VisitID", "24s", _clean_ascii),
                    ("VisitDate", "d", _parse_date),
                    ("GridType", "I", _get_first),
                    ("GridOffset", "I", _get_first),
                    ("GridType1", "I", _get_first),
                    ("GridOffset1", "I", _get_first),
                    ("ProgID", "34s", _clean_ascii),
                    ("__empty", "1790s", _get_first)], }

BSCAN_HDR_VERSIONS = {
    "HSF-BS-103": [("Version", "12s", _clean_ascii),
                   ("BScanHdrSize", "i", _get_first),
                   ("StartX", "d", _get_first),
                   ("StartY", "d", _get_first),
                   ("EndX", "d", _get_first),
                   ("EndY", "d", _get_first),
                   ("NumSeg", "I", _get_first),
                   ("OffSeg", "I", _get_first),
                   ("Quality", "f", _get_first),
                   ("Shift", "I", _get_first),
                   ("IVTrafo", "ffffff", lambda x: x)],
}


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
    def prop(self):
        field_position = self._meta_fields.index(field)
        startpos = self._startpos + calcsize(
            "=" + "".join(self._fmt[:field_position]))
        fmt = self._fmt[field_position]

        if getattr(self, f"_{field}") is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(startpos, 0)
                content = myfile.read(calcsize(fmt))
                attr = func(unpack(fmt, content))

                setattr(self, f"_{field}", attr)

        return getattr(self, f"_{field}")

    return property(prop)


class Oct:
    # def __new__(cls, bscans, slo, meta, *args, **kwargs):
    #    version = meta.Version
    #    for meta_attr in meta._meta_fields:
    #        setattr(cls, meta_attr, property(lambda self: getattr(self.meta, meta_attr)))
    #    return object.__new__(cls, *args, **kwargs)

    def __init__(self, bscans, slo, meta):
        self._bscans = bscans
        self._sloreader = slo
        self._meta = meta

        self._slo = None

    def __getattr__(self, attr):
        if attr in self.meta._meta_fields:
            return getattr(self.meta, attr)
        else:
            raise ValueError(f"'OCT' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return self._bscans[key]

    @property
    def segmentation(self):
        segmentations = np.stack([bscan.segmentation for bscan in self.bscans])
        # It seems like there is no standard structure in the exported segmentations from HEYEX
        # seg_mapping = {"ILM":0,"GCL":2, "BM":1, "IPl":3, "INL":4, "IPL":5, "ONL":6, "ELM":8, "EZ/PR1":14, "IZ/PR2":15,
        #               "RPE":16}
        # return {k: segmentations[:, seg_mapping[k], :] for k in seg_mapping}
        return {
            "{}".format(i): segmentations[:, i, :]
            for i in range(segmentations.shape[1])
        }

    @property
    def meta(self):
        return self._meta

    @property
    def bscan_meta(self):
        return [bs._meta for bs in self._bscans]

    @property
    def volume(self):
        return np.stack([x._scan for x in self._bscans], axis=-1)

    @property
    def slo(self):
        if self._slo is None:
            self._slo = self._sloreader.data
        return self._slo

    @property
    def bscans(self):
        return self._bscans

    @property
    def bscan_region(self):
        pass

    @classmethod
    def read_vol(cls, filepath):
        meta = get_octmeta(filepath)
        bscans = get_bscans(filepath, meta)
        slo = get_slo(filepath, meta)
        return cls(bscans, slo, meta)

    @classmethod
    def read_xml(cls, filepath):
        raise NotImplementedError("OCT.read_xml() is not implemented.")


class OctMeta:
    """
    The specification for the file header shown below was found in
    https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
    {'Version','c',0}, ... 		    % Version identifier: HSF-OCT-xxx, xxx = version number of the file format,
                                      Current version: xxx = 103
    {'SizeX','i',12},  ... 			% Number of A-Scans in each B-Scan, i.e. the width of each B-Scan in pixel
    {'NumBScans','i',16}, ... 		% Number of B-Scans in OCT scan
    {'SizeZ','i',20}, ... 			% Number of samples in an A-Scan, i.e. the Height of each B-Scan in pixel
    {'ScaleX','d',24}, ... 			% Width of a B-Scan pixel in mm
    {'Distance','d',32}, ... 		% Distance between two adjacent B-Scans in mm
    {'ScaleZ','d',40}, ... 			% Height of a B-Scan pixel in mm
    {'SizeXSlo','i',48}, ...  		% Width of the SLO image in pixel
    {'SizeYSlo','i',52}, ... 		% Height of the SLO image in pixel
    {'ScaleXSlo','d',56}, ... 		% Width of a pixel in the SLO image in mm
    {'ScaleYSlo','d',64}, ...		% Height of a pixel in the SLO image in mm
    {'FieldSizeSlo','i',72}, ... 	% Horizontal field size of the SLO image in dgr
    {'ScanFocus','d',76}, ...		% Scan focus in dpt
    {'ScanPosition','c',84}, ... 	% Examined eye (zero terminated string). "OS" for left eye; "OD" for right eye
    {'ExamTime','i',88}, ... 		% Examination time. The structure holds an unsigned 64-bit date and time value
                                      and represents the number of 100-nanosecond units since the beginning of
                                      January 1, 1601.
    {'ScanPattern','i',96}, ...		% Scan pattern type: 0 = Unknown pattern, 1 = Single line scan (one B-Scan
                                      only),
                                      2 = Circular scan (one B-Scan only), 3 = Volume scan in ART mode,
                                      4 = Fast volume scan, 5 = Radial scan (aka. star pattern)
    {'BScanHdrSize','i',100}, ...	% Size of the Header preceding each B-Scan in bytes
    {'ID','c',104}, ...				% Unique identifier of this OCT-scan (zero terminated string). This is
                                      identical to
                                      the number <SerID> that is part of the file name. Format: n[.m] n and m are
                                      numbers. The extension .m exists only for ScanPattern 1 and 2. Examples: 2390, 3433.2
    {'ReferenceID','c',120}, ...	% Unique identifier of the reference OCT-scan (zero terminated string). Format:
                                      see ID, This ID is only present if the OCT-scan is part of a progression otherwise
                                      this string is empty. For the reference scan of a progression ID and ReferenceID
                                      are identical.
    {'PID','i',136}, ...			% Internal patient ID used by HEYEX.
    {'PatientID','c',140}, ...		% User-defined patient ID (zero terminated string).
    {'Padding','c',161}, ...		% To align next member to 4-byte boundary.
    {'DOB','date',164}, ... 		% Patient's date of birth
    {'VID','i',172}, ...			% Internal visit ID used by HEYEX.
    {'VisitID','c',176}, ...		% User-defined visit ID (zero terminated string). This ID can be defined in the
                                      Comment-field of the Diagnosis-tab of the Examination Data dialog box. The VisitID
                                      must be defined in the first row of the comment field. It has to begin with an "#"
                                      and ends with any white-space character. It can contain up to 23 alpha-numeric
                                      characters (excluding the "#").
    {'VisitDate','date',200}, ...	% Date the visit took place. Identical to the date of an examination tab in HEYEX.
    {'GridType','i',208}, ...		% Type of grid used to derive thickness data. 0 No thickness data available,
                                      >0 Type of grid used to derive thickness  values. Seeter "Thickness Grid"	for more
                                      details on thickness data, Thickness data is only available for ScanPattern 3 and 4.
    {'GridOffset','i',212}, ...		% File offset of the thickness data in the file. If GridType is 0, GridOffset is 0.
    {'GridType1','i',216}, ...		% Type of a 2nd grid used to derive a 2nd set of thickness data.
    {'GridOffset1','i',220}, ...	% File offset of the 2 nd thickness data set in the file.
    {'ProgID','c',224}, ...			% Internal progression ID (zero terminated string). All scans of the same
                                      progression share this ID.
    {'Spare','c',258}}; 			% Spare bytes for future use. Initialized to 0.
    """

    def __new__(cls, filepath, version=None, *args, **kwargs):
        if version is None:
            with open(filepath, mode="rb") as myfile:
                fmt = "=12s"
                content = myfile.read(calcsize(fmt))
                version = unpack(fmt, content)[0].decode("ascii").rstrip("\x00")

        for k, v in _create_properties(version, OCT_HDR_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath):
        self._filepath = filepath
        self._startpos = 0

        for key in self._meta_fields[1:]:
            setattr(self, f"_{key}", None)

    def __str__(self):
        return f"{os.linesep}".join(
            [f"{f}: {getattr(self, f)}" for f in self._meta_fields
             if f != "__empty"])

    def __repr__(self):
        return self.__str__()
        # return f'OctMeta(filepath="{self._filepath}", version="{self.Version}")'


class Slo:
    def __init__(self, filepath, oct_meta=None):
        self._filepath = filepath
        self._data = None

        self._oct_meta = oct_meta

        self.shape = (self.oct_meta.SizeXSlo, self.oct_meta.SizeYSlo)
        self._size = self.shape[0] * self.shape[1]
        self._slo_start = 2048

    @property
    def data(self):
        if self._data is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(self._slo_start, 0)
                content = myfile.read(self._size)

            # Read SLO image
            fmt = f"={self._size}B"
            self._data = np.asarray(unpack(fmt, content),
                                    dtype="uint8").reshape(
                self.shape
            )

        return self._data

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._filepath)
        return self._oct_meta


class Bscan:
    def __new__(
        cls, filepath, startpos, oct_meta, *args, version="HSF-BS-103", **kwargs
    ):

        for k, v in _create_properties(version, BSCAN_HDR_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, startpos, oct_meta):
        """
        The specification of the B-scan header shown below was found in:
        https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
        {'Version','c',0}, ...          % Version identifier (zero terminated string). Version Format: "HSF-BS-xxx,
                                          xxx = version number of the B-Scan header format. Current version: xxx = 103
        {'BScanHdrSize','i',12}, ...    % Size of the B-Scan header in bytes. It is identical to the same value of the
                                          file header.
        {'StartX','d',16}, ...          % X-Coordinate of the B-Scan's start point in mm.
        {'StartY','d',24}, ...          % Y-Coordinate of the B-Scan's start point in mm.
        {'EndX','d',32}, ...            % X-Coordinate of the B-Scan's end point in mm. For circle scans, this is the
                                          X-Coordinate of the circle's center point.
        {'EndY','d',40}, ...            % Y-Coordinate of the B-Scan's end point in mm. For circle scans, this is the
                                          Y-Coordinate of the circle's center point.
        {'NumSeg','i',48}, ...          % Number of segmentation vectors
        {'OffSeg','i',52}, ...          % Offset of the array of segmentation vectors relative to the beginning of this
                                          B-Scan header.
        {'Quality','f',56}, ...         % Image quality measure. If this value does not exist, its value is set to
                                          INVALID.
        {'Shift','i',60}, ...           % Horizontal shift (in # of A-Scans) of the classification band against the
                                          segmentation lines (for circular scan only).
        {'IVTrafo','f',64}, ...         % Intra volume transformation matrix. The values are only available for volume
                                          and radial scans and if alignment is turned off, otherwise the values are
                                          initialized to 0.
        {'Spare','c',88}};              % Spare bytes for future use.
        """

        self._filepath = filepath
        self._startpos = startpos

        self.oct_meta = oct_meta
        self._meta = None
        self._scan = None
        self._segmentation = None

        for key in self._meta_fields[1:]:
            setattr(self, f"_{key}", None)

    @property
    def meta(self):
        if self._meta is None:
            if self._filepath is None or self._startpos is None:
                raise ValueError(
                    "meta is not set and filepath or B-Scan startpos is missing."
                )

            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(self._startpos, 0)
                content = myfile.read(self.oct_meta.BScanHdrSize)

            hdr_tail_size = self.oct_meta.BScanHdrSize - 68
            fmt = f"={''.join(self._fmt).replace('=', '')}{hdr_tail_size}s"
            bs_header = unpack(fmt, content)
            self._meta = {f: bh for f, bh in zip(self._meta_fields, bs_header)}

            for key in self._meta:
                setattr(self, f"_{key}", self._meta[key])
        return self._meta

    @property
    def Version(self):
        field_position = self._meta_fields.index("Version")
        startpos = self._startpos + calcsize(
            "=" + "".join(self._fmt[:field_position]))
        fmt = "=" + self._fmt[field_position]

        if self._Version is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(startpos, 0)
                content = myfile.read(calcsize(fmt))
                self._Version = unpack(fmt, content)[0].decode("ascii").rstrip(
                    "\x00")

        return self._Version

    @property
    def _segmentation_start(self):
        return self._hdr_start + self.OffSeg

    @property
    def _segmentation_size(self):
        return self.NumSeg * self.oct_meta.SizeX

    @property
    def segmentation(self):
        try:
            if self._segmentation is None:
                with open(self._filepath, mode="rb") as myfile:
                    myfile.seek(self._segmentation_start, 0)
                    content = myfile.read(self._segmentation_size * 4)

                f = f"{str(int(self._segmentation_size))}f"
                f = f"{self._segmentation_size}f"
                seg_lines = unpack(f, content)
                seg_lines = np.asarray(seg_lines, dtype="float32")
                return seg_lines.reshape(self.NumSeg, -1)
        except:
            return None

    @property
    def scan(self):
        if self._scan is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(self._scan_start, 0)
                content = myfile.read(self._scan_size * 4)

            f = str(int(self._scan_size)) + "f"
            bscan_img = unpack(f, content)
            bscan_img = np.asarray(bscan_img, dtype="float32")
            bscan_img[bscan_img > 1] = 0

            self._scan = bscan_img.reshape(self.oct_meta.SizeZ,
                                           self.oct_meta.SizeX)

        return self._scan

    @property
    def _scan_start(self):
        return self._startpos + self.oct_meta.BScanHdrSize

    @property
    def _scan_size(self):
        return self.oct_meta.SizeX * self.oct_meta.SizeZ


class Bscans:
    def __init__(self, filepath, oct_meta=None):
        self._filepath = filepath
        self._hdr_start = None
        self._oct_meta = oct_meta

    @property
    def hdr_start(self):
        if self._hdr_start is None:
            self.seek(0)
        return self._hdr_start

    @hdr_start.setter
    def hdr_start(self, value):
        self._hdr_start = value

    def seek(self, bscan_position):
        if bscan_position > self.oct_meta.NumBScans - 1:
            raise ValueError(
                f"There are only {self.oct_meta.NumBScans} B-Scans.")
        oct_header_size = 2048
        slo_size = self.oct_meta.SizeXSlo * self.oct_meta.SizeYSlo
        bscan_size = self.oct_meta.SizeX * self.oct_meta.SizeZ
        self.hdr_start = (
            oct_header_size
            + slo_size
            + bscan_position * (4 * bscan_size)
            + ((bscan_position) * self.oct_meta.BScanHdrSize)
        )

    @property
    def hdr_size(self):
        return self.oct_meta.BScanHdrSize

    def _get_current_bscan(self):
        return Bscan(self._filepath, self.hdr_start, oct_meta=self.oct_meta)

    def __len__(self):
        return self.oct_meta.NumBScans

    def __getitem__(self, sli):
        if type(sli) == int:
            if sli < 0:
                sli = self.oct_meta.NumBScans + sli
            elif sli >= self.oct_meta.NumBScans:
                raise IndexError
            self.seek(sli)
            return self._get_current_bscan()
        else:
            if sli.start < 0:
                sli.start = self.oct_meta.NumBScans + sli.start
            if sli.end < 0:
                sli.end = self.oct_meta.NumBScans + sli.end
            bscans = []
            if sli.step is None:
                step = 1
            else:
                step = sli.step
            for i in range(sli.start, sli.stop, step):
                self.seek(i)
                bscans.append(self._get_current_bscan())
            return bscans

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._filepath)
        return self._oct_meta
