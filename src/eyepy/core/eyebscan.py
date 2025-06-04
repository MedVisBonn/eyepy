from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from eyepy import config
from eyepy.core.annotations import EyeBscanLayerAnnotation
from eyepy.core.annotations import EyeBscanSlabAnnotation
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.plotting import plot_scalebar
from eyepy.core.plotting import plot_watermark
from eyepy.core.utils import DynamicDefaultDict

if TYPE_CHECKING:
    from eyepy import EyeVolume



class EyeBscan:
    """"""

    def __init__(self, volume: EyeVolume, index: int) -> None:
        """

        Args:
            volume: The EyeVolume this B-scan belongs to
            index: The index of this B-scan in the EyeVolume
        """
        self.index = index
        self.volume = volume

        # Create a dict to access layers by their name for this B-scan.
        self.layers = DynamicDefaultDict(lambda x: EyeBscanLayerAnnotation(
            self.volume.layers[x], self.index))
        self.area_maps = DynamicDefaultDict(
            lambda x: self.volume.volume_maps[x].data[self.index])
        self.slabs = DynamicDefaultDict(lambda x: EyeBscanSlabAnnotation(
            self.volume.slabs[x], self.index))

    @property
    def meta(self) -> EyeBscanMeta:
        """Return the metadata for this B-scan.

        Returns:
            Meta information about the B-scan
        """
        return self.volume.meta['bscan_meta'][self.index]

    @property
    def data(self) -> np.ndarray:
        """Returns the B-scan data as a numpy array.

        Returns:
            B-scan data as numpy array
        """
        return self.volume.data[self.index]

    #@property
    #def ascan_maps(self):
    #    """

    #    Returns:

    #    """
    #    raise NotImplementedError
    # return self.volume.ascan_maps[self.index]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the B-scan data.

        Returns:
            Shape tuple (B-scan height, B-scan width)
        """
        return self.data.shape

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        layers: Union[bool, list[str]] = False,
        areas: Union[bool, list[str]] = False,
        slabs: Union[bool, list[str]] = False,
        #ascans=None,
        layer_kwargs: Optional[dict] = None,
        area_kwargs: Optional[dict] = None,
        slab_kwargs: Optional[dict] = None,
        #ascan_kwargs=None,
        annotations_only: bool = False,
        region: tuple[slice, slice] = np.s_[:, :],
        scalebar: Union[bool, str] = 'botleft',
        scalebar_kwargs: Optional[dict[str, Any]] = None,
        watermark: bool = True,
    ) -> None:
        """Plot B-scan.

        Annotations such as layers and areas can be overlaid on the image. With plt.legend() you can add a legend for the shown annotations

        Args:
            ax: Axes to plot on. If not provided plot on the current axes (plt.gca()).
            layers: If `True` plot all layers (default: `False`). If a list of strings is given, plot the layers with the given names.
            areas: If `True` plot all areas (default: `False`). If a list of strings is given, plot the areas with the given names.
            slabs: If `True` plot all slabs (default: `False`). If a list of strings is given, plot the slabs with the given names.
            annotations_only: If `True` do not plot the B-scan image
            region: Region of the localizer to plot (default: `np.s_[:, :]`)
            layer_kwargs: Optional keyword arguments for customizing the OCT layers. If `None` default values are used which are {"linewidth": 1, "linestyle": "-"}
            area_kwargs: Optional keyword arguments for customizing area annotions on the B-scan If `None` default values are used which are {"alpha": 0.5}
            slab_kwargs: Optional keyword arguments for customizing slab annotations on the B-scan If `None` default values are used which are {"alpha": 0.5}
            scalebar: Position of the scalebar, one of "topright", "topleft", "botright", "botleft" or `False` (default: "botleft"). If `True` the scalebar is placed in the bottom left corner. You can custumize the scalebar using the `scalebar_kwargs` argument.
            scalebar_kwargs: Optional keyword arguments for customizing the scalebar. Check the documentation of [plot_scalebar][eyepy.core.plotting.plot_scalebar] for more information.
            watermark: If `True` plot a watermark on the image (default: `True`). When removing the watermark, please consider to cite eyepy in your publication.
        Returns:
            None
        """
        ax = plt.gca() if ax is None else ax

        # Complete region index expression
        y_start = region[0].start if region[0].start is not None else 0
        y_stop = region[0].stop if region[0].stop is not None else self.shape[0]
        x_start = region[1].start if region[1].start is not None else 0
        x_stop = region[1].stop if region[1].stop is not None else self.shape[1]

        region = np.s_[y_start:y_stop, x_start:x_stop]

        if not layers:
            layers = []
        elif layers is True:
            layers = list(self.volume.layers.keys())

        if not areas:
            areas = []
        elif areas is True:
            areas = list(self.volume.volume_maps.keys())

        if not slabs:
            slabs = []
        elif slabs is True:
            slabs = list(self.volume.slabs.keys())
            # Exclude 'RET' slab from the list of slabs to plot
            slabs = [s for s in slabs if s != 'RET']

        #if ascans is None:
        #    ascans = []
        #elif ascans is True:
        #    ascans = self.ascan_maps.keys()

        if layer_kwargs is None:
            layer_kwargs = config.layer_kwargs
        else:
            layer_kwargs = {**config.layer_kwargs, **layer_kwargs}

        if area_kwargs is None:
            area_kwargs = config.area_kwargs
        else:
            area_kwargs = {**config.area_kwargs, **area_kwargs}

        if slab_kwargs is None:
            slab_kwargs = config.slab_kwargs
        else:
            slab_kwargs = {**config.slab_kwargs, **slab_kwargs}

        #if ascan_kwargs is None:
        #    ascan_kwargs = config.area_kwargs
        #else:
        #    ascan_kwargs = {**config.ascan_kwargs, **ascan_kwargs}

        if not annotations_only:
            ax.imshow(self.data[region], cmap='gray')

        #for ascan_annotation in ascans:
        #    data = self.ascan_maps[ascan_annotation]
        #    data = np.repeat(np.reshape(data, (1, -1)), self.shape[0], axis=0)
        #    visible = np.zeros(data.shape)
        #    visible[data] = 1.0
        #    ax.imshow(data[region],
        #              alpha=visible[region] * ascan_kwargs["alpha"],
        #              cmap="Reds")

        for area in areas:
            data = self.area_maps[area][region]
            visible = np.zeros(data.shape, dtype=bool)
            visible[data != 0] = 1.0

            meta = self.volume.volume_maps[area].meta
            color = meta['color'] if 'color' in meta else 'red'
            color = mcolors.to_rgba(color)
            # create a 0 radius circle patch as dummy for the area label
            patch = mpatches.Circle((0, 0), radius=0, color=color, label=area)
            ax.add_patch(patch)

            # Create plot_data by tiling the color vector over the plotting shape
            plot_data = np.tile(np.array(color), data.shape + (1, ))
            # Now turn the alpha channel 0 where the mask is 0 and adjust the remaining alpha
            plot_data[..., 3] *= visible * area_kwargs['alpha']

            ax.imshow(
                plot_data,
                interpolation='none',
            )
        for layer in layers:
            color = config.layer_colors[layer]

            layer_data = self.layers[layer].data
            # Adjust layer height to plotted region
            layer_data = layer_data - region[0].start
            # Remove layer if outside of region
            layer_data = layer_data[region[1].start:region[1].stop]
            layer_data[layer_data < 0] = 0
            region_height = region[0].stop - region[0].start
            layer_data[layer_data > region_height] = region_height

            ax.plot(
                layer_data,
                color='#' + color,
                label=layer,
                **layer_kwargs,
            )
        if slabs:
            # Create a composite RGB image
            composite = np.zeros((*self.data[region].shape[:2], 3))
            overlap_count = np.zeros(self.data[region].shape[:2])

            for slab in slabs:
                color = config.slab_colors[slab]
                color_rgb = mcolors.to_rgb('#' + color)

                slab_mask = self.slabs[slab].mask
                slab_mask = slab_mask[region]

                # Add to composite where mask is True
                mask_indices = slab_mask > 0
                composite[mask_indices, :] += np.array(color_rgb)
                overlap_count[mask_indices] += 1

            # Normalize by the actual number of overlapping slabs per pixel
            valid_pixels = overlap_count > 0
            composite[valid_pixels, :] /= overlap_count[valid_pixels, np.newaxis]

            alpha = slab_kwargs.pop('alpha', 0.5)

            # Add alpha channel to composite
            composite_with_alpha = np.zeros((*composite.shape[:2], 4))
            composite_with_alpha[..., :3] = composite
            composite_with_alpha[..., 3] = valid_pixels * alpha

            # pop alpha from slab_kwargs since we already added it to the composite
            slab_kwargs.pop('alpha', None)
            ax.imshow(composite_with_alpha,
                      **slab_kwargs)

        # Make sure tick labels match the image region
        y_start = region[0].start if region[0].start is not None else 0
        x_start = region[1].start if region[1].start is not None else 0
        y_end = region[0].stop if region[0].stop is not None else self.shape[0]
        x_end = region[1].stop if region[1].stop is not None else self.shape[1]

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

            scale_unit = self.volume.meta['scale_unit']
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

    @property
    def size_x(self):
        """Size of the B-scan in x direction."""
        return self.shape[1]

    @property
    def size_y(self):
        """Size of the B-scan in y direction."""
        return self.shape[0]

    @property
    def scale_x(self):
        """Scale of the B-scan in x direction."""
        return self.volume.scale_x

    @property
    def scale_y(self):
        """Scale of the B-scan in y direction."""
        return self.volume.scale_y
