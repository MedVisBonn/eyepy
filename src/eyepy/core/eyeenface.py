from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np

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

    def plot(self,
             ax: Optional[plt.Axes] = None,
             region: Union[slice, Tuple[slice, slice]] = np.s_[:, :]):
        """

        Args:
            ax:
            region:

        Returns:

        """
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.data[region], cmap="gray")

        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())

        y_start = region[0].start if region[0].start is not None else 0
        x_start = region[1].start if region[1].start is not None else 0

        ax.set_yticklabels((ax.get_yticks() + y_start).astype(int))
        ax.set_xticklabels((ax.get_xticks() + x_start).astype(int))
