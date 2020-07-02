import functools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio
import numpy as np

from .const import HEXML_VERSIONS, HEXML_BSCAN_VERSIONS, SEG_MAPPING
from .utils import _get_meta_attr, _create_properties_xml

"""
Inspired by:
https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""


@functools.lru_cache(maxsize=4, typed=False)
def get_xml_root(filepath):
    tree = ET.parse(filepath)
    return tree.getroot()


@functools.lru_cache(maxsize=4, typed=False)
def get_slo(filepath, oct_meta=None):
    return HeyexSlo(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_bscans(filepath, oct_meta=None):
    return HeyexBscans(filepath, oct_meta)


@functools.lru_cache(maxsize=4, typed=False)
def get_octmeta(filepath):
    return HeyexOctMeta(filepath)


class HeyexOctMeta:
    def __new__(cls, filepath, version=None, *args, **kwargs):
        if version is None:
            root = get_xml_root(filepath)
            version = root[0].find("SWVersion")[1].text

        for k, v in _create_properties_xml(version, HEXML_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath):
        """

        Parameters
        ----------
        file_obj :
        """
        self._filepath = filepath
        self._root = get_xml_root(filepath)

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
        """

        Parameters
        ----------
        root :
        oct_meta :
        """
        self._xmlfilepath = Path(filepath)
        self._root = get_xml_root(filepath)
        self._data = None

        self._oct_meta = oct_meta

        self.shape = (self.oct_meta.SizeXSlo, self.oct_meta.SizeYSlo)
        self._size = self.shape[0] * self.shape[1]

    @property
    def data(self):
        if self._data is None:
            localizer_path = self._root[0].find(
                ".//ImageType[Type='LOCALIZER']../ImageData/ExamURL").text
            localizer_name = localizer_path.split("\\")[-1]
            self._data = imageio.imread(
                self._xmlfilepath.parent / localizer_name)
        return self._data

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._xmlfilepath)
        return self._oct_meta


class HeyexBscanMeta:
    def __new__(cls, filepath, bscan_index, version=None, *args, **kwargs):
        if version is None:
            root = get_xml_root(filepath)
            version = root[0].find("SWVersion")[1].text

        for k, v in _create_properties_xml(version,
                                           HEXML_BSCAN_VERSIONS).items():
            setattr(cls, k, v)
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, bscan_index, *args, **kwargs):
        """

        Parameters
        ----------
        filepath :
        startpos :
        """
        self._xmlfilepath = Path(filepath)
        self._root = get_xml_root(filepath)[0].findall(f".//Image[ImageNumber='{bscan_index+1}']")

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
        cls, filepath, bscan_index, oct_meta=None, version=None, *args,
        **kwargs):
        meta = HeyexBscanMeta(filepath, bscan_index, version)
        setattr(cls, "meta", meta)

        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, bscan_index, oct_meta):
        """

        Parameters
        ----------
        filepath :
        bscan_index :
        oct_meta :
        """
        self._xmlfilepath = Path(filepath)
        self._index = bscan_index
        self._root = get_xml_root(filepath)[0].findall(
            ".//ImageType[Type='OCT']..")[self._index]
        
        self._index = bscan_index

        self.oct_meta = oct_meta
        self._scan = None
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
            self._segmentation_raw = np.full(shape=(17, self.oct_meta.SizeX),
                                             fill_value=np.nan, dtype="float32")

            for segline in self._root.findall(".//SegLine"):
                name = segline.find("./Name").text
                self._segmentation_raw[SEG_MAPPING[name], :] = \
                    [float(x) for x in segline.find("./Array").text.split()]
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
        if self._scan is None:
            img_path = self._root.find("./ImageData/ExamURL").text
            img_name = img_path.split("\\")[-1]
            self._scan = imageio.imread(self._xmlfilepath.parent / img_name)
        return self._scan

    @property
    def scan_raw(self):
        raise NotImplementedError("The Heyex XML export does not contain the"
                                  " raw OCT data.")

    @property
    def size(self):
        return self.oct_meta.SizeX * self.oct_meta.SizeZ


class HeyexBscans:
    def __init__(self, filepath, oct_meta=None):
        """

        Parameters
        ----------
        filepath :
        oct_meta :
        """
        self._xmlfilepath = Path(filepath)
        self._root = get_xml_root(filepath)
        self._oct_meta = oct_meta

    def _get_bscan(self, index):
        return HeyexBscan(self._xmlfilepath, index,
                          oct_meta=self.oct_meta)

    def __len__(self):
        return self.oct_meta.NumBScans

    def __getitem__(self, index):
        if type(index) == int:
            if index < 0:
                index = self.oct_meta.NumBScans + index
            elif index >= self.oct_meta.NumBScans:
                raise IndexError
            return self._get_bscan(index)
        elif type(index) == slice:
            bscans = []
            for i in range(*index.indices(self.oct_meta.NumBScans)):
                bscans.append(self._get_bscan(i))
            return bscans
        else:
            raise TypeError("Index has to be of type 'int' or 'slice'")

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            self._oct_meta = get_octmeta(self._xmlfilepath)
        return self._oct_meta
