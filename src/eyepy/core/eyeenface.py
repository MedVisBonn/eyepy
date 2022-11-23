from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt

from eyepy.core.annotations import EyeEnfaceAreaAnnotation


class EyeEnface:
    """ """

    def __init__(self, data, meta):
        """

        Args:
            data:
            meta:
        """
        self.data = data
        self._area_maps = []
        self.meta = meta

    @property
    def area_maps(self):
        """

        Returns:

        """
        # Create a dict to access area_maps by their name
        return {am.name: am for am in self._area_maps}

    def add_area_annotation(self, area_map=None, meta=None, **kwargs):
        """

        Args:
            area_map:
            meta:
            **kwargs:

        Returns:

        """
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        area_annotation = EyeEnfaceAreaAnnotation(self, area_map, meta)
        self._area_maps.append(area_annotation)
        return area_annotation

    @property
    def scale_x(self):
        """

        Returns:

        """
        return self.meta["scale_x"]

    @property
    def scale_y(self):
        """

        Returns:

        """
        return self.meta["scale_y"]

    @property
    def size_x(self):
        """

        Returns:

        """
        return self.shape[1]

    @property
    def size_y(self):
        """

        Returns:

        """
        return self.shape[0]

    @property
    def laterality(self):
        """

        Returns:

        """
        return self.meta["laterality"]

    @property
    def shape(self):
        """

        Returns:

        """
        return self.data.shape

    def plot(self, ax=None, region=np.s_[...]):
        """

        Args:
            ax:
            region:

        Returns:

        """
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.data[region], cmap="gray")
