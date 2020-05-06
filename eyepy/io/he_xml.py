# -*- coding: utf-8 -*-
import numpy as np
import untangle


class SpectralisXMLReader(object):
    def __init__(self, path):
        self.obj = untangle.parse(path)

    @property
    def version(self):
        self.obj.HEDX.BODY.SWVersion.Version.cdata

    @property
    def octs(self):
        images = self.obj.HEDX.BODY.Patient.Study.Series.Image
        return [i for i in images if i.ImageType.Type == "OCT"]

    @property
    def oct_filenames(self):
        octs = self.octs
        return self.get_oct_filenames(octs)

    @property
    def localizer_filename(self):
        return self.get_localizer_filename()

    @property
    def oct_width(self):
        return int(self.octs[0].OphthalmicAcquisitionContext.Width.cdata)

    @property
    def oct_height(self):
        return int(self.octs[0].OphthalmicAcquisitionContext.Height.cdata)

    @property
    def oct_n(self):
        return int(len(self.octs))

    def get_segmentations(self):

        self.segmentations = {}
        seg_types = [x.Name for x in self.octs[0].Segmentation.SegLine]

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
            for s in self.octs[i].Segmentation.SegLine:
                if s.Name.cdata == name:
                    seg_data = np.array(s.Array.cdata.split(" ")).astype("float")
                    x = np.where(seg_data < 10000)
                    y = np.rint(seg_data[x]).astype(int)
                    seg[i, y, x] = 1
                    # seg = (seg*255).astype(np.uint8)

        return seg

    def get_oct_filenames(self, octs):
        names = []
        for o in octs:
            names.append(o.ImageData.ExamURL.cdata.split("\\")[-1])
        return names

    def get_localizer_filename(self):
        images = self.obj.HEDX.BODY.Patient.Study.Series.Image
        path = [i for i in images if i.ImageType.Type == "LOCALIZER"][0]
        return path.ImageData.ExamURL.cdata.split("\\")[-1]
