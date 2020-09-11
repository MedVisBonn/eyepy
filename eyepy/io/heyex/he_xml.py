import functools
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio
import numpy as np

from eyepy.core.config import SEG_MAPPING
from eyepy.core.octbase import Bscan
from eyepy.io.utils import _get_meta_attr, _create_properties_xml
from .specification.xml_export import HEXML_VERSIONS, HEXML_BSCAN_VERSIONS


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


class HeyexBscan(Bscan):
    def __new__(
        cls, filepath, bscan_index, oct_volume, oct_meta=None, version=None,
        *args, **kwargs):
        meta = HeyexBscanMeta(filepath, bscan_index, version)
        setattr(cls, "meta", meta)

        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, filepath, bscan_index, oct_volume, oct_meta):
        """

        Parameters
        ----------
        filepath :
        bscan_index :
        oct_meta :
        """
        super().__init__(bscan_index, oct_volume)
        self._xmlfilepath = Path(filepath)
        self._root = get_xml_root(filepath)[0].findall(
            ".//ImageType[Type='OCT']..")[self._index]

        self.oct_meta = oct_meta
        self._scan = None
        self._segmentation_raw = None

        self.scan_name = \
        self._root.find("./ImageData/ExamURL").text.split("\\")[-1]

    @property
    def shape(self):
        return (self.oct_meta.SizeZ, self.oct_meta.SizeX)

    @property
    def _segmentation_start(self):
        return self._startpos + self.OffSeg

    @property
    def _segmentation_size(self):
        return self.oct_meta.SizeX * 17

    @property
    def layers_raw(self):
        if self._segmentation_raw is None:
            seglines = self._root.findall(".//SegLine")
            self._segmentation_raw = np.full(shape=(17, self.oct_meta.SizeX),
                                             fill_value=np.nan, dtype="float32")
            if seglines:
                for segline in seglines:
                    name = segline.find("./Name").text
                    self._segmentation_raw[SEG_MAPPING[name], :] = \
                        [float(x) for x in segline.find("./Array").text.split()]
            else:
                warnings.warn(f"{self._xmlfilepath.stem} contains no segmentation",
                                    UserWarning)

        return self._segmentation_raw

    @property
    def layers(self):
        data = self.layers_raw.copy()
        nans = np.isnan(self.layers_raw)
        empty = np.nonzero(np.logical_or(
            np.less(self.layers_raw, 0, where=~nans),
            np.greater(self.layers_raw, self.oct_meta.SizeZ, where=~nans)))
        data[empty] = np.nan
        return {name: data[i] for name, i in SEG_MAPPING.items()
                if np.nansum(data[i]) != 0}


    @property
    def scan(self):
        if self._scan is None:
            self._scan = imageio.imread(self._xmlfilepath.parent / self.scan_name)[..., 0]
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
        self._oct_volume = None

    def _get_bscan(self, index):
        return HeyexBscan(self._xmlfilepath, index,
                          oct_meta=self.oct_meta, oct_volume=self._oct_volume)

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
