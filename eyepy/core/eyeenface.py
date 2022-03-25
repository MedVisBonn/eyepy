from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt


class EyeEnfaceAreaAnnotation:
    def __init__(
        self,
        enface: "EyeEnface",
        data: Optional[npt.NDArray[bool]] = None,
        meta: Optional[dict] = None,
        **kwargs
    ):
        self.enface = enface

        if data is None:
            self.data = np.full(self.enface.shape, fill_value=False, dtype=bool)
        else:
            self.data = data

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        if "name" not in self.meta:
            self.meta["name"] = "Pixel Annotation"

    @property
    def name(self):
        return self.meta["name"]

    @name.setter
    def name(self, value):
        self.meta["name"] = value


class EyeEnface:
    def __init__(self, data, meta):
        self.data = data
        self._area_maps = []
        self.meta = meta

    @property
    def area_maps(self):
        # Create a dict to access area_maps by their name
        return {am.name: am for am in self._area_maps}

    def add_area_annotation(self, area_map=None, meta=None, **kwargs):
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        area_annotation = EyeEnfaceAreaAnnotation(self, area_map, meta)
        self._area_maps.append(area_annotation)
        return area_annotation

    @property
    def scale_x(self):
        return self.meta["scale_x"]

    @property
    def scale_y(self):
        return self.meta["scale_y"]

    @property
    def size_x(self):
        return self.shape[1]

    @property
    def size_y(self):
        return self.shape[0]

    @property
    def laterality(self):
        return self.meta["laterality"]

    @property
    def shape(self):
        return self.data.shape

    def plot(self, ax=None, region=np.s_[...]):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.data[region], cmap="gray")

    def register(self):
        pass
