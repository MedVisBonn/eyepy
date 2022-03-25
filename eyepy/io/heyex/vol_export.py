# -*- coding: utf-8 -*-
"""Inspired by:

https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""

import logging
import mmap
from pathlib import Path, PosixPath
from struct import calcsize, unpack
from typing import IO, Union

import numpy as np
from skimage import img_as_ubyte

from eyepy.io.lazy import (
    LazyAnnotation,
    LazyBscan,
    LazyEnfaceImage,
    LazyLayerAnnotation,
    LazyMeta,
)
from eyepy.io.utils import _clean_ascii

from .specification.vol_export import HEVOL_BSCAN_VERSIONS, HEVOL_VERSIONS

logger = logging.getLogger(__name__)


class HeyexVolReader:
    """A reader for HEYEX .vol exports.

    This reader lazy loads a .vol file. It gives you access to the B-Scans with
    their annotations and meta data, the localizer image and the OCTs meta data.

    Attributes:
        bscans: A list of functions. Every function returns a 'Bscan' object
            when called
        localizer: An 'EnfaceImage' object
        oct_meta: A 'Meta' object.
    """

    def __init__(self, file_obj: Union[str, Path, IO], version=None):

        if type(file_obj) is str or type(file_obj) is PosixPath:
            with open(file_obj, "rb") as myfile:
                self.memmap = mmap.mmap(myfile.fileno(), 0, access=mmap.ACCESS_READ)
                self.path = Path(file_obj).parent
        else:
            self.memmap = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            self.path = Path(file_obj.name).parent

        if version is None:
            fmt = "=12s"
            content = self.memmap.read(calcsize(fmt))
            version = _clean_ascii(unpack(fmt, content))
        self.version = version
        self.bscan_version = version.replace("OCT", "BS")

        self._bscans = None
        self._localizer = None
        self._oct_meta = None

    @property
    def bscans(self):
        if self._bscans is None:
            oct_header_size = 2048
            slo_size = self.oct_meta["SizeXSlo"] * self.oct_meta["SizeYSlo"]
            bscan_size = self.oct_meta["SizeX"] * self.oct_meta["SizeY"]
            shape = (self.oct_meta["SizeY"], self.oct_meta["SizeX"])

            def bscan_builder(d, a, bmeta, p):
                return lambda: LazyBscan(d, a, bmeta, p)

            self._bscans = []
            for index in range(self.oct_meta["NumBScans"]):
                startpos = (
                    oct_header_size
                    + slo_size
                    + index * (4 * bscan_size)
                    + ((index) * self.oct_meta["BScanHdrSize"])
                )
                data = np.ndarray(
                    buffer=self.memmap,
                    dtype="float32",
                    offset=startpos + self.oct_meta["BScanHdrSize"],
                    shape=shape,
                )

                bscan_meta = LazyMeta(
                    **self.create_meta_retrieve_funcs_heyex_vol(
                        HEVOL_BSCAN_VERSIONS(self.bscan_version), startpos
                    )
                )

                annotation = LazyAnnotation(**self.create_annotation_dict(startpos))

                self._bscans.append(
                    bscan_builder(data, annotation, bscan_meta, self._data_processing)
                )

        return self._bscans

    @property
    def localizer(self):
        if self._localizer is None:
            shape = (self.oct_meta["SizeXSlo"], self.oct_meta["SizeYSlo"])
            self._localizer = LazyEnfaceImage(
                data=np.ndarray(
                    buffer=self.memmap, dtype="uint8", offset=2048, shape=shape
                )
            )
        return self._localizer

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            retrieve_dict = self.create_meta_retrieve_funcs_heyex_vol(
                HEVOL_VERSIONS(self.version)
            )
            self._oct_meta = LazyMeta(**retrieve_dict)
        return self._oct_meta

    def _data_processing(self, data):
        """How to process the loaded B-Scans."""
        data = np.copy(data)
        data[data > 1.1] = 0.0
        # return data
        func = lambda x: np.rint(
            (np.log(np.clip(x, 3.8e-06, 0.99) + 2.443e-04) + 8.301) * 1.207e-01 * 255
        )
        return img_as_ubyte(func(data).astype(int))

    def create_annotation_dict(self, startpos):
        """For every Annotation create a function to read it.

        Currently only a function to read the layer segmentaton is
        returned.
        """

        def layers_dict(bscan_obj):
            data = np.ndarray(
                buffer=self.memmap,
                dtype="float32",
                offset=startpos + bscan_obj.OffSeg,
                shape=(17, bscan_obj.oct_obj.SizeX),
            )
            return LazyLayerAnnotation(data, max_height=bscan_obj.oct_obj.SizeY)

        return {
            "layers": layers_dict,
        }

    def create_meta_retrieve_funcs_heyex_vol(self, specification, offset=0):
        """For every meta field, create a function to read it from the file.

        Return all functions in a dict name: func
        """
        fields = [entry[0] for entry in specification]
        fmts = [entry[1] for entry in specification]
        funcs = [entry[2] for entry in specification]

        def func_builder(fnctn, frmt, startpos):
            def retrieve_func():
                self.memmap.seek(startpos, 0)
                content = self.memmap.read(calcsize(frmt))
                return fnctn(unpack(frmt, content))

            return retrieve_func

        func_dict = {}
        for index, (field, func, fmt) in enumerate(zip(fields, funcs, fmts)):
            startpos = offset + calcsize("=" + "".join(fmts[:index]))
            func_dict[field] = func_builder(func, fmt, startpos)

        return func_dict
