# -*- coding: utf-8 -*-
import numpy as np
import untangle
from pathlib import Path
import imageio

def read_xml_export(path):
    return SpectralisXMLReader(path)

class SpectralisXMLReader(object):
    def __init__(self, path):
        self.path = Path(path)
        self.xml_path = list(self.path.glob("*.xml"))
        print(self.xml_path)
        if len(self.xml_path) != 1:
            raise ValueError("There is not exactly one .xml file under the given path.")
        else:
            self.xml_path = self.xml_path[0]
        self.xml_obj = untangle.parse(str(self.xml_path))

        self._slo = None
        self._slo_filename = None
        self._bscan_filenames = None
        self._NumBScans = None

    @property
    def slo(self):
        if self._slo is None:
            self._slo = imageio.imread(self.path / self.slo_filename)
        return self._slo

    def _get_bscan(self, index):
        return imageio.imread(self.path / self.bscan_filenames[index])


    def __getitem__(self, index):
        if type(index) == int:
            if index < 0:
                index = self.NumBScans + index
            elif index >= self.NumBScans:
                raise IndexError
            return self._get_bscan(index)
        elif type(index) == slice:
            bscans = []
            for b in range(*index.indices(self.NumBScans)):
                bscans.append(self._get_bscan(b))
            return bscans
        else:
            raise TypeError("Index has to be of type 'int' or 'slice'")

    @property
    def Version(self):
        self.xml_obj.HEDX.BODY.SWVersion.Version.cdata

    @property
    def NumBScans(self):
        if self._NumBScans is None:
            self._NumBScans = len(self.bscans())
        return self._NumBScans

    def bscans(self):
        images = self.xml_obj.HEDX.BODY.Patient.Study.Series.Image
        return [i for i in images if i.ImageType.Type == "OCT"]

    @property
    def bscan_filenames(self):
        if self._bscan_filenames is None:
            self._bscan_filenames = self.get_bscan_filenames(self.bscans())
        return self._bscan_filenames

    @property
    def slo_filename(self):
        if self._slo_filename is None:
            images = self.xml_obj.HEDX.BODY.Patient.Study.Series.Image
            path = [i for i in images if i.ImageType.Type == "LOCALIZER"][0]
            self._slo_filename =  path.ImageData.ExamURL.cdata.split("\\")[-1]

        return self._slo_filename

    @property
    def oct_width(self):
        return int(self.bscans[0].OphthalmicAcquisitionContext.Width.cdata)

    @property
    def oct_height(self):
        return int(self.bscans[0].OphthalmicAcquisitionContext.Height.cdata)

    @property
    def oct_n(self):
        return int(len(self.bscans))

    def get_segmentations(self):

        self.segmentations = {}
        seg_types = [x.Name for x in self.bscans[0].Segmentation.SegLine]

        for seg_type in seg_types:
            self.segmentations[seg_type.cdata] = self.get_segmentation(seg_type.cdata)

            self.segmentations[seg_type.cdata] *= 255
            self.segmentations[seg_type.cdata] = self.segmentations[
                seg_type.cdata
            ].astype(np.uint8)
        return self.segmentations

    def get_segmentation(self, name):
        seg = np.zeros((self.oct_n, self.oct_height, self.oct_width))
        for i in range(self.oct_n):
            for s in self.bscans[i].Segmentation.SegLine:
                if s.Name.cdata == name:
                    seg_data = np.array(s.Array.cdata.split(" ")).astype("float")
                    x = np.where(seg_data < 10000)
                    y = np.rint(seg_data[x]).astype(int)
                    seg[i, y, x] = 1
                    # seg = (seg*255).astype(np.uint8)

        return seg

    def get_bscan_filenames(self, octs):
        names = []
        for o in octs:
            names.append(o.ImageData.ExamURL.cdata.split("\\")[-1])
        return names
