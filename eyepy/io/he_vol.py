import functools
import os
from pathlib import Path
from struct import unpack, calcsize
from typing import Union

import numpy as np
from skimage import img_as_ubyte

from .const import OCT_HDR_VERSIONS, BSCAN_HDR_VERSIONS
from .utils import _get_meta_attr, _create_properties, _clean_ascii

"""
Inspired by:
https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""


@functools.lru_cache(maxsize=4, typed=False)
def get_slo(filepath, oct_meta=None):
    return HeyexSlo(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_bscans(filepath, oct_meta=None):
    return HeyexBscans(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_octmeta(filepath):
    return HeyexOctMeta(filepath)


def read_vol(filepath: Union[str, Path]):
    """ Read a Heyex OCT from a .vol file and return a HeyexOct object.

    Parameters
    ----------
    filepath : Path to the .vol file

    Returns
    -------
    HeyexOct
    """
    return HeyexOct.read_vol(filepath)


class HeyexOct:
    """


    The HeyexOct object has not yet read the .vol file when returned to you. It
    will only read exactly what you ask for. No B-Scan image is read from the
    file if you only want the SLO or a specific field from the .vol header. This
    makes it faster to analyse large collections of .vol files based on their
    headers.

    .vol header
    -----------
    All fields from the .vol header can be accessed as attributes of the
    HeyexOct.  object:

    Accessing the attribute `slo` reads the IR SLO image and returns it as a
    numpy.ndarray of dtype `uint8`.

    Individual B-Scans can be accessed using `oct_scan[index]`. The returned
    HeyexBscan object exposes all B-Scan header fields as attributes and the
    raw B-Scan image as `numpy.ndarray` of type `float32` under the attribute
    `scan_raw`.

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
    def segmentation(self):
        segmentations = np.stack([bscan.segmentation for bscan in self._bscans])
        # It seems like there is no standard structure in the exported segmentations from HEYEX
        # seg_mapping = {"ILM":0,"GCL":2, "BM":1, "IPl":3, "INL":4, "IPL":5, "ONL":6, "ELM":8, "EZ/PR1":14, "IZ/PR2":15,
        #               "RPE":16}
        # return {k: segmentations[:, seg_mapping[k], :] for k in seg_mapping}
        return {
            "{}".format(i): segmentations[:, i, :]
            for i in range(segmentations.shape[1])
        }

    @property
    def volume(self):
        return np.stack([x.scan_raw for x in self.bscans], axis=-1)

    @classmethod
    def read_vol(cls, filepath):
        meta = get_octmeta(filepath)
        bscans = get_bscans(filepath, meta)
        slo = get_slo(filepath, meta)
        return cls(bscans, slo, meta)


class HeyexOctMeta:
    """

    """

    def __new__(cls, filepath, version=None, *args, **kwargs):
        if version is None:
            with open(filepath, mode="rb") as myfile:
                fmt = "=12s"
                content = myfile.read(calcsize(fmt))
                version = _clean_ascii(unpack(fmt, content))

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


class HeyexSlo:
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


class HeyexBscanMeta:
    def __new__(cls, filepath, startpos, version=None, *args, **kwargs):
        if version is None:
            with open(filepath, mode="rb") as myfile:
                fmt = "=12s"
                myfile.seek(startpos)
                content = myfile.read(calcsize(fmt))
                version = _clean_ascii(unpack(fmt, content))

        for k, v in _create_properties(version, BSCAN_HDR_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, startpos, version=None):
        self._filepath = filepath
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
        cls, filepath, startpos, oct_meta=None, version=None, *args, **kwargs):
        meta = HeyexBscanMeta(filepath, startpos, version)
        setattr(cls, "meta", meta)

        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, startpos, oct_meta):
        self._filepath = filepath
        self._startpos = startpos

        self.oct_meta = oct_meta
        self._scan_raw = None
        self._segmentation = None

    @property
    def _segmentation_start(self):
        return self._startpos + self.OffSeg

    @property
    def _segmentation_size(self):
        return self.NumSeg * self.oct_meta.SizeX

    @property
    def segmentation(self):
        if self._segmentation is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(self._segmentation_start, 0)
                content = myfile.read(self._segmentation_size * 4)

            f = f"{str(int(self._segmentation_size))}f"
            f = f"{self._segmentation_size}f"
            seg_lines = unpack(f, content)
            seg_lines = np.asarray(seg_lines, dtype="float32")
            return seg_lines.reshape(self.NumSeg, -1)

    @property
    def scan(self):
        return img_as_ubyte(np.power(self.scan_raw, 1 / 4))

    @property
    def scan_raw(self):
        if self._scan_raw is None:
            with open(self._filepath, mode="rb") as myfile:
                myfile.seek(self._scan_start, 0)
                content = myfile.read(self._scan_size * 4)

            f = str(int(self._scan_size)) + "f"
            bscan_img = unpack(f, content)
            bscan_img = np.asarray(bscan_img, dtype="float32")

            # Ignore regions masked by max float
            bscan_img[bscan_img > 1.1] = 0

            self._scan_raw = bscan_img.reshape(self.oct_meta.SizeZ,
                                               self.oct_meta.SizeX)

        return self._scan_raw

    @property
    def _scan_start(self):
        return self._startpos + self.oct_meta.BScanHdrSize

    @property
    def _scan_size(self):
        return self.oct_meta.SizeX * self.oct_meta.SizeZ


class HeyexBscans:
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
        return HeyexBscan(self._filepath, self.hdr_start,
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
            self._oct_meta = get_octmeta(self._filepath)
        return self._oct_meta
