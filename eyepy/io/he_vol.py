import functools
import mmap
import os
from pathlib import Path, PosixPath
from struct import unpack, calcsize
from typing import Union, IO

import numpy as np
from skimage import img_as_ubyte

from .const import OCT_HDR_VERSIONS, BSCAN_HDR_VERSIONS, SEG_MAPPING
from .utils import _get_meta_attr, _create_properties, _clean_ascii

"""
Inspired by:
https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""


@functools.lru_cache(maxsize=4, typed=False)
def get_slo(file_obj, oct_meta=None):
    return HeyexSlo(file_obj, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_bscans(file_obj, oct_meta=None):
    return HeyexBscans(file_obj, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_octmeta(file_obj):
    return HeyexOctMeta(file_obj)


def read_vol(file_obj: Union[str, Path, IO]):
    """ Return a HeyexOct object for a .vol file at the given file_obj

    Parameters
    ----------
    file_obj : Path to the .vol file

    Returns
    -------
    HeyexOct
    """

    if type(file_obj) is str or type(file_obj) is PosixPath:
        with open(file_obj, "rb") as myfile:
            mm = mmap.mmap(myfile.fileno(), 0, access=mmap.ACCESS_READ)
            return HeyexOct.read_vol(mm)
    else:
        mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        return HeyexOct.read_vol(mm)


class HeyexOct:
    """
    The HeyexOct object lazy loads the .vol file. It will only read exactly what
    you ask for. This means that no B-Scan image is read from the file if you
    only want the SLO or a specific field from the .vol header.
    In comparison to reading the complete file, this makes all operations on
    .vol files faster which do not require the complete file e.g. peaking at
    header fields, plotting the SLO, plotting individual B-Scans.

    .vol header
    -----------
    All fields from the .vol header can be accessed as attributes of the
    HeyexOct object.

    SLO
    ---
    The attribute `slo` of the HeyexOct object gives access to the IR SLO image
    and returns it as a numpy.ndarray of dtype `uint8`.

    B-Scans
    -------
    Individual B-Scans can be accessed using `oct_scan[index]`. The returned
    HeyexBscan object exposes all B-Scan header fields as attributes and the
    raw B-Scan image as `numpy.ndarray` of type `float32` under the attribute
    `scan_raw`. A transformed version of the raw B-Scan which is more similar to
    the Heyex experience can be accessed with the attribute `scan` and returns
    the 4th root of the raw B-Scan scaled to [0,255] as `uint8`.

    Segmentations
    -------------
    B-Scan segmentations can be accessed for individual B-Scans like
    `bscan.segmentation`. This return a numpy.ndarray of shape (NumSeg, SizeX)
    The `segmentation` attribute of the HeyexOct object returns a dictionary,
    where the key is a number and the value is a numpy.ndarray of shape
    (NumBScans, SizeX).

    """

    def __new__(cls, bscans, slo, meta, *args, **kwargs):
        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, bscans, slo, meta):
        """

        Parameters
        ----------
        bscans :
        slo :
        meta :
        """
        self._bscans = bscans
        self._sloreader = slo
        self._meta = meta

        self._slo = None
        self._segmentation_raw = None

    def __getitem__(self, key):
        return self._bscans[key]

    @property
    def slo(self):
        if self._slo is None:
            self._slo = self._sloreader.data
        return self._slo

    @property
    def meta(self):
        return self._meta

    @property
    def segmentation_raw(self):
        if self._segmentation_raw is None:
            self._segmentation_raw = np.stack([x.segmentation_raw
                                               for x in self._bscans], axis=1)
        return self._segmentation_raw

    @property
    def segmentation(self):
        empty = np.nonzero(np.logical_or(
            self.segmentation_raw < 0,
            self.segmentation_raw > self.meta.SizeZ)
        )
        data = self.segmentation_raw.copy()
        data[empty] = np.nan
        return {name: data[i, ...] for name, i in SEG_MAPPING.items()
                if np.nansum(data[i, ...]) != 0}

    @property
    def volume_raw(self):
        return np.stack([x.scan_raw for x in self._bscans], axis=-1)

    @property
    def volume(self):
        return np.stack([x.scan for x in self._bscans], axis=-1)

    @classmethod
    def read_vol(cls, file_obj):
        meta = get_octmeta(file_obj)
        bscans = get_bscans(file_obj, meta)
        slo = get_slo(file_obj, meta)
        return cls(bscans, slo, meta)


class HeyexOctMeta:
    def __new__(cls, file_obj, version=None, *args, **kwargs):
        if version is None:
            fmt = "=12s"
            content = file_obj.read(calcsize(fmt))
            version = _clean_ascii(unpack(fmt, content))

        for k, v in _create_properties(version, OCT_HDR_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, file_obj):
        """

        Parameters
        ----------
        file_obj :
        """
        self._file_obj = file_obj
        self._startpos = 0

        for key in self._meta_fields[1:]:
            setattr(self, f"_{key}", None)

    def __str__(self):
        return f"{os.linesep}".join(
            [f"{f}: {getattr(self, f)}" for f in self._meta_fields
             if f != "__empty"])

    def __repr__(self):
        return self.__str__()


class HeyexSlo:
    def __init__(self, file_obj, oct_meta=None):
        """

        Parameters
        ----------
        file_obj :
        oct_meta :
        """
        self._file_obj = file_obj
        self._data = None

        self._oct_meta = oct_meta

        self.shape = (self.oct_meta.SizeXSlo, self.oct_meta.SizeYSlo)
        self._size = self.shape[0] * self.shape[1]
        self._slo_start = 2048

    @property
    def data(self):
        if self._data is None:
            self._data = np.ndarray(buffer=self._file_obj, dtype="uint8",
                                    offset=self._slo_start, shape=self.shape)
        return self._data

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._file_obj)
        return self._oct_meta


class HeyexBscanMeta:
    def __new__(cls, file_obj, startpos, version=None, *args, **kwargs):
        if version is None:
            fmt = "=12s"
            file_obj.seek(startpos)
            content = file_obj.read(calcsize(fmt))
            version = _clean_ascii(unpack(fmt, content))

        for k, v in _create_properties(version, BSCAN_HDR_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, file_obj, startpos, *args, **kwargs):
        """

        Parameters
        ----------
        file_obj :
        startpos :
        """
        self._file_obj = file_obj
        self._startpos = startpos

        for key in self._meta_fields[1:]:
            setattr(self, f"_{key}", None)

    def __str__(self):
        return f"{os.linesep}".join(
            [f"{f}: {getattr(self, f)}" for f in self._meta_fields
             if f != "__empty"])

    def __repr__(self):
        return self.__str__()


class HeyexBscan:
    def __new__(
        cls, file_obj, startpos, oct_meta=None, version=None, *args, **kwargs):
        meta = HeyexBscanMeta(file_obj, startpos, version)
        setattr(cls, "meta", meta)

        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, file_obj, startpos, oct_meta):
        """

        Parameters
        ----------
        file_obj :
        startpos :
        oct_meta :
        """
        self._file_obj = file_obj
        self._startpos = startpos

        self.oct_meta = oct_meta
        self._scan_raw = None
        self._segmentation_raw = None

    @property
    def _segmentation_start(self):
        return self._startpos + self.OffSeg

    @property
    def _segmentation_size(self):
        return self.oct_meta.SizeX * 17

    @property
    def segmentation_raw(self):
        if self._segmentation_raw is None:
            self._segmentation_raw = np.ndarray(buffer=self._file_obj,
                                                dtype="float32",
                                                offset=self._segmentation_start,
                                                shape=(17, self.oct_meta.SizeX))
        return self._segmentation_raw

    @property
    def segmentation(self):
        data = self.segmentation_raw.copy()
        empty = np.nonzero(np.logical_or(data < 0, data > self.oct_meta.SizeZ))
        data[empty] = np.nan
        return {name: data[i] for name, i in SEG_MAPPING.items()
                if np.nansum(data[i]) != 0}

    @property
    def scan(self):
        # Ignore regions masked by max float
        data = self.scan_raw.copy()
        data[data > 1.1] = np.nan
        return img_as_ubyte(np.power(data, 1 / 4))

    @property
    def scan_raw(self):
        if self._scan_raw is None:
            shape = (self.oct_meta.SizeZ, self.oct_meta.SizeX)
            self._scan_raw = np.ndarray(buffer=self._file_obj, dtype="float32",
                                        offset=self._scan_start,
                                        shape=shape)
        return self._scan_raw

    @property
    def _scan_start(self):
        return self._startpos + self.oct_meta.BScanHdrSize

    @property
    def _scan_size(self):
        return self.oct_meta.SizeX * self.oct_meta.SizeZ


class HeyexBscans:
    def __init__(self, file_obj, oct_meta=None):
        """

        Parameters
        ----------
        file_obj :
        oct_meta :
        """
        self._file_obj = file_obj
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
        return HeyexBscan(self._file_obj, self.hdr_start,
                          oct_meta=self.oct_meta)

    def __len__(self):
        return self.oct_meta.NumBScans

    def __getitem__(self, index):
        if type(index) == int:
            if index < 0:
                index = self.oct_meta.NumBScans + index
            elif index >= self.oct_meta.NumBScans:
                raise IndexError
            self.seek(index)
            return self._get_current_bscan()
        elif type(index) == slice:
            bscans = []
            for b in range(*index.indices(self.oct_meta.NumBScans)):
                self.seek(b)
                bscans.append(self._get_current_bscan())
            return bscans
        else:
            raise TypeError("Index has to be of type 'int' or 'slice'")

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._file_obj)
        return self._oct_meta
