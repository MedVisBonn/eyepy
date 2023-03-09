from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np

from eyepy.core.annotations import EyeEnfacePixelAnnotation


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
        area_annotation = EyeEnfacePixelAnnotation(self, area_map, meta)
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

        # Make sure tick labels match the image region
        y_start = region[0].start if region[0].start is not None else 0
        x_start = region[1].start if region[1].start is not None else 0
        y_end = region[0].stop if region[0].stop is not None else self.size_y
        x_end = region[1].stop if region[1].stop is not None else self.size_x

        # Ticks are not clipped to the image region. Clip them here.
        yticks = ax.get_yticks()
        yticks = yticks[np.nonzero(
            np.logical_and(yticks >= 0, yticks <= y_end - y_start - 1))]
        xticks = ax.get_xticks()
        xticks = xticks[np.nonzero(
            np.logical_and(xticks >= 0, xticks <= x_end - x_start - 1))]

        # Set clipped ticks (this is only necessary because we change the labels later)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)

        # Set labels to ticks + start of the region as an offset
        ax.set_yticklabels([str(int(t + y_start)) for t in yticks])
        ax.set_xticklabels([str(int(t + x_start)) for t in xticks])
