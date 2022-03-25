# -*- coding: utf-8 -*-
import functools
import logging
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import imageio
import numpy as np
from skimage import img_as_ubyte

from eyepy.io.heyex.specification.xml_export import HEXML_BSCAN_VERSIONS, HEXML_VERSIONS
from eyepy.io.lazy import (
    SEG_MAPPING,
    LazyAnnotation,
    LazyBscan,
    LazyEnfaceImage,
    LazyLayerAnnotation,
    LazyMeta,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4, typed=False)
def get_xml_root(filepath):
    tree = ElementTree.parse(filepath)
    return tree.getroot()


class HeyexXmlReader:
    """A reader for HEYEX .xml exports.

    This reader lazy loads a .xml export. It gives you access to B-Scans with
    their annotations and meta data, the localizer image and the OCTs meta data.

    Attributes:
        bscans: A list of functions. Every function returns a 'Bscan' object
            when called
        localizer: An 'EnfaceImage' object
        oct_meta: A 'Meta' object.
    """

    def __init__(self, path):
        path = Path(path)
        if not path.suffix == ".xml":
            xmls = list(path.glob("*.xml"))
            if len(xmls) == 0:
                raise FileNotFoundError(
                    "There is no .xml file under the given filepath"
                )
            elif len(xmls) > 1:
                raise ValueError(
                    "There is more than one .xml file in the given folder."
                )
            path = xmls[0]
        self.path = path
        self.xml_root = get_xml_root(self.path)
        self.version = self.xml_root[0].find("SWVersion")[1].text

        self._oct_meta = None

    @property
    def bscans(self):
        bscans = []

        def bscan_builder(d, a, bmeta, p, n):
            return lambda: LazyBscan(d, a, bmeta, p, name=n)

        def scan_reader(path):
            return lambda: imageio.imread(path)

        for bscan in self.xml_root[0].findall(".//ImageType[Type='OCT'].."):
            scan_name = bscan.find("./ImageData/ExamURL").text.split("\\")[-1]
            data = scan_reader(self.path.parent / scan_name)
            annotation = LazyAnnotation(**self.create_annotation_dict(bscan))
            bscan_meta = LazyMeta(
                **self.create_meta_retrieve_funcs_heyex_xml(
                    bscan, HEXML_BSCAN_VERSIONS(self.version)
                )
            )

            bscans.append(
                bscan_builder(
                    data, annotation, bscan_meta, self._data_processing, scan_name
                )
            )

        return bscans

    @property
    def localizer(self):
        localizer_pattern = ".//ImageType[Type='LOCALIZER']../ImageData/ExamURL"
        localizer_name = self.xml_root[0].find(localizer_pattern).text.split("\\")[-1]
        return LazyEnfaceImage(
            data=lambda: imageio.imread(self.path.parent / localizer_name),
            name=localizer_name,
        )

    @property
    def oct_meta(self):
        if self._oct_meta is None:
            retrieve_dict = self.create_meta_retrieve_funcs_heyex_xml(
                self.xml_root[0], HEXML_VERSIONS(self.version)
            )
            self._oct_meta = LazyMeta(**retrieve_dict)

        return self._oct_meta

    def _data_processing(self, data):
        """How to process the loaded B-Scans."""
        if data.ndim == 3:
            return img_as_ubyte(data[..., 0])
        return img_as_ubyte(data)

    def create_meta_retrieve_funcs_heyex_xml(self, xml_root, specification):
        """For every meta field, create a function to read it from the file.

        Return all functions in a dict name: func
        """
        fields = [entry[0] for entry in specification]
        locs = [entry[1] for entry in specification]
        funcs = [entry[2] for entry in specification]

        func_dict = {}

        def func_builder(xml, f, lo):
            return lambda: f(xml.findall(lo))

        for field, func, loc in zip(fields, funcs, locs):
            func_dict[field] = func_builder(xml_root, func, loc)

        return func_dict

    def create_annotation_dict(self, bscan_root):
        """For every Annotation create a function to read it.

        Currently only a function to read the layer segmentaton is
        returned.
        """

        def layers_dict(bscan_obj):

            seglines = bscan_root.findall(".//SegLine")

            if seglines:
                data = np.full(
                    shape=(17, bscan_obj.oct_obj.SizeX),
                    fill_value=np.nan,
                    dtype="float32",
                )
                for segline in seglines:
                    name = segline.find("./Name").text
                    data[SEG_MAPPING[name], :] = [
                        float(x) for x in segline.find("./Array").text.split()
                    ]
                return LazyLayerAnnotation(data, max_height=bscan_obj.oct_obj.SizeY)
            else:
                # warnings.warn(f"{bscan_obj} contains no segmentation", UserWarning)
                data = np.zeros(
                    (max(SEG_MAPPING.values()) + 1, bscan_obj.oct_obj.SizeX)
                )
                return LazyLayerAnnotation(data, max_height=bscan_obj.oct_obj.SizeY)

        return {
            "layers": layers_dict,
        }
