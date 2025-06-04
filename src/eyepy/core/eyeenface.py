from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np

from eyepy.core.annotations import EyeEnfacePixelAnnotation
from eyepy.core.plotting import plot_scalebar
from eyepy.core.plotting import plot_watermark

if TYPE_CHECKING:
    from eyepy import EyeEnfaceMeta


class EyeEnface:
    """"""

    def __init__(self, data: npt.NDArray[np.int64],
                 meta: 'EyeEnfaceMeta') -> None:
        """

        Args:
            data:
            meta:
        """
        self.data = data
        self._area_maps = []
        self.meta = meta

    @property
    def area_maps(self) -> dict[str, EyeEnfacePixelAnnotation]:
        """

        Returns:

        """
        # Create a dict to access area_maps by their name
        return {am.name: am for am in self._area_maps}

    def add_area_annotation(self,
                            area_map: Optional[npt.NDArray[np.bool_]] = None,
                            meta: Optional[dict] = None,
                            **kwargs: Any) -> EyeEnfacePixelAnnotation:
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
    def scale_x(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_x']

    @property
    def scale_y(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_y']

    @property
    def size_x(self) -> int:
        """

        Returns:

        """
        return self.shape[1]

    @property
    def size_y(self) -> int:
        """

        Returns:

        """
        return self.shape[0]

    @property
    def laterality(self) -> str:
        """

        Returns:

        """
        return self.meta['laterality']

    @property
    def shape(self) -> tuple[int, int]:
        """

        Returns:

        """
        return self.data.shape

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        region: tuple[slice, slice] = np.s_[:, :],
        scalebar: Union[bool, str] = 'botleft',
        scalebar_kwargs: Optional[dict[str, Any]] = None,
        watermark: bool = True,
    ) -> None:
        """

        Args:
            ax: Axes to plot on. If not provided plot on the current axes (plt.gca()).
            region: Region of the localizer to plot (default: `np.s_[:, :]`)
            scalebar: Position of the scalebar, one of "topright", "topleft", "botright", "botleft" or `False` (default: "botleft"). If `True` the scalebar is placed in the bottom left corner. You can custumize the scalebar using the `scalebar_kwargs` argument.
            scalebar_kwargs: Optional keyword arguments for customizing the scalebar. Check the documentation of [plot_scalebar][eyepy.core.plotting.plot_scalebar] for more information.
            watermark: If `True` plot a watermark on the image (default: `True`). When removing the watermark, please consider to cite eyepy in your publication.
        Returns:
            None

        """
        ax = plt.gca() if ax is None else ax
        vmin = np.min(self.data)
        vmax = np.max(self.data)

        ax.imshow(self.data[region], cmap='gray', vmin=vmin, vmax=vmax)

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

        if scalebar:
            if scalebar_kwargs is None:
                scalebar_kwargs = {}

            scale_unit = self.meta['scale_unit']
            scalebar_kwargs = {
                **{
                    'scale': (self.scale_x, self.scale_y),
                    'scale_unit': scale_unit
                },
                **scalebar_kwargs
            }

            if not 'pos' in scalebar_kwargs:
                sx = x_end - x_start
                sy = y_end - y_start

                if scalebar is True:
                    scalebar = 'botleft'

                if scalebar == 'botleft':
                    scalebar_kwargs['pos'] = (sx - 0.95 * sx, 0.95 * sy)
                elif scalebar == 'botright':
                    scalebar_kwargs['pos'] = (0.95 * sx, 0.95 * sy)
                    scalebar_kwargs['flip_x'] = True
                elif scalebar == 'topleft':
                    scalebar_kwargs['pos'] = (sx - 0.95 * sx, 0.05 * sy)
                    scalebar_kwargs['flip_y'] = True
                elif scalebar == 'topright':
                    scalebar_kwargs['pos'] = (0.95 * sx, 0.05 * sy)
                    scalebar_kwargs['flip_x'] = True
                    scalebar_kwargs['flip_y'] = True

            plot_scalebar(ax=ax, **scalebar_kwargs)

        if watermark:
            plot_watermark(ax)
