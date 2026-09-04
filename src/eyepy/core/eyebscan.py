from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

import numpy as np

import eyepy.config as epconfig
from eyepy.core.annotations import EyeBscanLayerAnnotation
from eyepy.core.annotations import EyeBscanSlabAnnotation
from eyepy.core.utils import DynamicDefaultDict

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from eyepy.core.eyemeta import EyeBscanMeta
    from eyepy.core.eyevolume import EyeVolume


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

        The intensity transform is applied only to this single B-scan slice,
        avoiding the computation of the transform for the entire volume.

        Returns:
            B-scan data as numpy array
        """
        return self.volume.intensity_transform(np.copy(self.volume._raw_data[self.index]))

    @property
    def raw_data(self) -> np.ndarray:
        """Returns a copy of the raw (untransformed) B-scan data.

        This provides fast access to the original data without applying
        any intensity transform. Returns a copy to prevent accidental
        modification of the underlying volume data.

        Returns:
            Copy of raw B-scan data as numpy array
        """
        return np.copy(self.volume._raw_data[self.index])

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

        Shape of raw data to avoid computation of intensity transform.

        Returns:
            Shape tuple (B-scan height, B-scan width)
        """
        return self.raw_data.shape

    def plot(
        self,
        ax: Axes | None = None,
        layers: bool | list[str] = False,
        areas: bool | list[str] = False,
        slabs: bool | list[str] = False,
        layer_labels: list[str] | None = None,
        area_labels: list[str] | None = None,
        #ascans=None,
        layer_kwargs: dict | None = None,
        area_kwargs: dict | None = None,
        slab_kwargs: dict | None = None,
        #ascan_kwargs=None,
        annotations_only: bool = False,
        region: tuple[slice, slice] = np.s_[:, :],
        scalebar: bool | str = 'botleft',
        scalebar_kwargs: dict[str, Any] | None = None,
        watermark: bool = True,
        autofocus: bool | int = False,
    ) -> None:
        """Plot B-scan.

        Annotations such as layers and areas can be overlaid on the image. With plt.legend() you can add a legend for the shown annotations

        Args:
            ax: Axes to plot on. If not provided plot on the current axes (plt.gca()).
            layers: If `True` plot all layers (default: `False`). If a list of strings is given, plot the layers with the given names.
            areas: If `True` plot all areas (default: `False`). If a list of strings is given, plot the areas with the given names.
            slabs: If `True` plot all slabs (default: `False`). If a list of strings is given, plot the slabs with the given names.
            layer_labels: Optional list of strings to label the layers in the legend. If `None` the layer names are used.
            area_labels: Optional list of strings to label the areas in the legend. If `None` the area names are used.
            annotations_only: If `True` do not plot the B-scan image
            region: Region of the localizer to plot (default: `np.s_[:, :]`)
            layer_kwargs: Optional keyword arguments for customizing the OCT layers. If `None` default values are used which are {"linewidth": 1, "linestyle": "-"}
            area_kwargs: Optional keyword arguments for customizing area annotions on the B-scan If `None` default values are used which are {"alpha": 0.5}
            slab_kwargs: Optional keyword arguments for customizing slab annotations on the B-scan If `None` default values are used which are {"alpha": 0.5}
            scalebar: Position of the scalebar, one of "topright", "topleft", "botright", "botleft" or `False` (default: "botleft"). If `True` the scalebar is placed in the bottom left corner. You can custumize the scalebar using the `scalebar_kwargs` argument.
            scalebar_kwargs: Optional keyword arguments for customizing the scalebar. Check the documentation of [plot_scalebar][eyepy.core.plotting.plot_scalebar] for more information.
            watermark: If `True` plot a watermark on the image (default: `True`). When removing the watermark, please consider to cite eyepy in your publication.
            autofocus: If a boolean and `True` crop the B-scan to the retina region. If an integer, center the B-scan on the retina and crop to the specified height. (default: `False`)
        Returns:
            None
        """
        from eyepy.core._compat import require_matplotlib
        mcolors = require_matplotlib('colors')
        mpatches = require_matplotlib('patches')
        plt = require_matplotlib('pyplot')

        ax = plt.gca() if ax is None else ax

        # Complete region index expression
        y_start = region[0].start if region[0].start is not None else 0
        y_stop = region[0].stop if region[0].stop is not None else self.shape[0]
        x_start = region[1].start if region[1].start is not None else 0
        x_stop = region[1].stop if region[1].stop is not None else self.shape[1]

        region = np.s_[y_start:y_stop, x_start:x_stop]

        if autofocus:
            from eyepy.quant.segmentation import compute_retina_mask
            mask = compute_retina_mask(self.data)
            # sum over columns
            mask_sum = np.sum(mask, axis=1)
            # find first and last index where mask sum is > 0
            indices = np.where(mask_sum > 0)[0]
            if len(indices) > 0:
                min_index = indices[0]
                max_index = indices[-1]

                if isinstance(autofocus, bool):
                    # crop to retina region
                    y_start = max(0, min_index - 10)
                    y_stop = min(self.shape[0], max_index + 10)
                elif isinstance(autofocus, int):
                    # center on retina
                    center = (min_index + max_index) // 2
                    half_height = autofocus // 2
                    y_start = max(0, center - half_height)
                    y_stop = min(self.shape[0], center + half_height)

                region = np.s_[y_start:y_stop, x_start:x_stop]


        if not layers:
            layers = []
        elif layers is True:
            layers = list(self.volume.layers.keys())
        elif isinstance(layers, str):
            layers = [layers]

        if not areas:
            areas = []
        elif areas is True:
            areas = list(self.volume.volume_maps.keys())
        elif isinstance(areas, str):
            areas = [areas]

        if not slabs:
            slabs = []
        elif slabs is True:
            slabs = list(self.volume.slabs.keys())
            # Exclude 'RET' slab from the list of slabs to plot
            slabs = [s for s in slabs if s != 'RET']
        elif isinstance(slabs, str):
            slabs = [slabs]

        #if ascans is None:
        #    ascans = []
        #elif ascans is True:
        #    ascans = self.ascan_maps.keys()

        if layer_kwargs is None:
            layer_kwargs = epconfig.layer_kwargs
        else:
            layer_kwargs = {**epconfig.layer_kwargs, **layer_kwargs}

        if area_kwargs is None:
            area_kwargs = epconfig.area_kwargs
        else:
            area_kwargs = {**epconfig.area_kwargs, **area_kwargs}

        if slab_kwargs is None:
            slab_kwargs = epconfig.slab_kwargs
        else:
            slab_kwargs = {**epconfig.slab_kwargs, **slab_kwargs}

        #if ascan_kwargs is None:
        #    ascan_kwargs = epconfig.area_kwargs
        #else:
        #    ascan_kwargs = {**epconfig.ascan_kwargs, **ascan_kwargs}

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

        for i, area in enumerate(areas):
            data = self.area_maps[area][region]
            visible = np.zeros(data.shape, dtype=bool)
            visible[data != 0] = 1.0

            meta = self.volume.volume_maps[area].meta
            color = meta['color'] if 'color' in meta else 'red'
            color = mcolors.to_rgba(color)
            # create a 0 radius circle patch as dummy for the area label
            label = area_labels[i] if area_labels else area
            patch = mpatches.Circle((0, 0), radius=0, color=color, label=label)
            ax.add_patch(patch)

            # Create plot_data by tiling the color vector over the plotting shape
            plot_data = np.tile(np.array(color), data.shape + (1, ))
            # Now turn the alpha channel 0 where the mask is 0 and adjust the remaining alpha
            plot_data[..., 3] *= visible * area_kwargs['alpha']

            ax.imshow(
                plot_data,
                interpolation='none',
            )
        for i, layer in enumerate(layers):
            color = epconfig.layer_colors[layer]

            layer_data = self.layers[layer].data
            # Adjust layer height to plotted region
            layer_data = layer_data - region[0].start
            # Remove layer if outside of region
            layer_data = layer_data[region[1].start:region[1].stop]
            layer_data[layer_data < 0] = 0
            region_height = region[0].stop - region[0].start
            layer_data[layer_data > region_height] = region_height

            label = layer_labels[i] if layer_labels else layer
            ax.plot(
                layer_data,
                color='#' + color,
                label=label,
                **layer_kwargs,
            )
        if slabs:
            # Create a composite RGB image
            composite = np.zeros((*self.data[region].shape[:2], 3))
            overlap_count = np.zeros(self.data[region].shape[:2])

            for slab in slabs:
                color = epconfig.slab_colors[slab]
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

            from eyepy.core.plotting import plot_scalebar
            plot_scalebar(ax=ax, **scalebar_kwargs)

        if watermark:
            from eyepy.core.plotting import plot_watermark
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
