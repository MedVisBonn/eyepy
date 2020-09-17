import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import cm, colors, patches
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import transform

from eyepy.core import config
from eyepy.core.drusen import DefaultDrusenFinder
from eyepy.core.quantifier import DefaultEyeQuantifier
from eyepy.io.utils import _get_meta_attr


class Oct(ABC):

    def __new__(cls, bscans, slo, meta, *args, **kwargs):
        # Set all the meta fields as attributes
        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    @abstractmethod
    def __init__(self, bscans, enfacereader, meta,
                 drusenfinder=DefaultDrusenFinder(),
                 eyequantifier=DefaultEyeQuantifier()):
        """

        Parameters
        ----------
        bscans :
        enfacereader :
        meta :
        drusenfinder :
        """
        self._bscans = bscans
        self._enfacereader = enfacereader
        self._enface = None
        self._meta = meta
        self._drusenfinder = drusenfinder
        self._eyequantifier = eyequantifier

        self._drusen = None
        self._drusen_raw = None
        self._segmentation_raw = None
        self._tform_enface_to_oct = None

        # Set the oct volume in the Bscans object to self such that every
        # loaded B-Scan can refer to the volume
        self._bscans._oct_volume = self

        # Try to estimate B-Scan distances if not given
        if self.Distance is None:
            # Pythagoras in case B-Scans are rotated with respect to the enface
            a = self[-1].StartY - self[0].StartY
            b = self[-1].StartX - self[0].StartX
            self.meta._Distance = np.sqrt(a ** 2 + b ** 2) / (
                    self.NumBScans - 1)

    def __getitem__(self, key):
        """ The B-Scan at the given index """
        return self._bscans[key]

    def __len__(self):
        """ The number of B-Scans """
        return len(self._bscans)

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def fovea_pos(self):
        """ Position of the Fovea on the localizer image """
        raise NotImplementedError()

    @property
    def enface(self):
        """ A numpy array holding the OCTs localizer enface if available """
        if self._enface is None:
            if not self._enfacereader is None:
                self._enface = self._enfacereader.data
            else:
                raise ValueError("There is no localizer enface available"
                                 " for this OCT")
        return self._enface

    @property
    def volume_raw(self):
        """ An array holding the OCT volume

        The dtype is not changed after the import. If available this is the
        unprocessed output of the OCT device. In any case this is the
        unprocessed data imported by eyepy.
        """
        return np.stack([x.scan_raw for x in self._bscans], axis=-1)

    @property
    def volume(self):
        """ An array holding a single B-Scan with the commonly used contrast

        The array is of dtype <ubyte> and encodes the intensities as values
        between 0 and 255.
        """
        return np.stack([x.scan for x in self._bscans], axis=-1)

    @property
    def layers_raw(self):
        """ Height maps for all layers combined into one volume.

        Layers for all B-Scans are stacked such that we get a volume L x B x W
        where L are different Layers, B are the B-Scans and W is the Width of
        the B-Scans.
        """
        if self._segmentation_raw is None:
            self._segmentation_raw = np.stack([x.layers_raw
                                               for x in self._bscans], axis=1)
        return self._segmentation_raw

    @property
    def layers(self):
        """ Height maps for all layers accessible by the layers name """
        nans = np.isnan(self.layers_raw)
        empty = np.nonzero(np.logical_or(
            np.less(self.layers_raw, 0, where=~nans),
            np.greater(self.layers_raw, self.meta.SizeZ, where=~nans)))

        data = self.layers_raw.copy()
        data[empty] = np.nan
        return {name: data[i, ...] for name, i in config.SEG_MAPPING.items()
                if np.nansum(data[i, ...]) != 0}

    @property
    def meta(self):
        """ A python object holding all OCT meta data as attributes

        The object can be printed to see all available meta data.
        """
        return self._meta

    @property
    def drusen(self):
        """ Final drusen after post processing the initial raw drusen

        Here the `filter` function of the DrusenFinder has been applied
        """
        if self._drusen is None:
            self._drusen = self._drusenfinder.filter(self.drusen_raw)
        return self._drusen

    @property
    def drusen_raw(self):
        """ Initial drusen before post procssing

        The initial drusen found by the DrusenFinders `find` function.
        """
        if self._drusen_raw is None:
            self._drusen_raw = self._drusenfinder.find(self)
        return self._drusen_raw

    @property
    def quantification(self):
        return self._eyequantifier.quantify(self)

    @property
    def tform_enface_to_oct(self):
        if self._tform_enface_to_oct is None:
            self._tform_enface_to_oct = self._estimate_enface_to_oct_tform()
        return self._tform_enface_to_oct

    @property
    def tform_oct_to_enface(self):
        return self.tform_enface_to_oct.inverse

    def _estimate_enface_to_oct_tform(self):
        dren_shape = self.drusen_projection.shape
        src = np.array(
            [dren_shape[0] - 1, 0,  # Top left
             dren_shape[0] - 1, dren_shape[1] - 1,  # Top right
             0, 0,  # Bottom left
             0, dren_shape[1] - 1  # Bottom right
             ]).reshape((-1, 2))

        dst = np.array(
            [self[-1].StartY / self.ScaleXSlo, self[-1].StartX / self.ScaleYSlo,
             self[-1].EndY / self.ScaleXSlo, self[-1].EndX / self.ScaleYSlo,
             self[0].StartY / self.ScaleXSlo, self[0].StartX / self.ScaleYSlo,
             self[0].EndY / self.ScaleXSlo, self[0].EndX / self.ScaleYSlo
             ]).reshape((-1, 2))

        src = src[:, [1, 0]]
        dst = dst[:, [1, 0]]
        tform = transform.estimate_transform("affine", src, dst)

        if not np.allclose(tform.inverse(tform(src)), src):
            msg = f"Problem with transformation of OCT Projection to SLO space."
            raise ValueError(msg)

        return tform

    @property
    def drusen_projection(self):
        return np.swapaxes(np.sum(self.drusen, axis=0), 0, 1)

    @property
    def drusen_enface(self):
        """ Drusen projection warped into the enface space """
        return transform.warp(self.drusen_projection.astype(float),
                              self.tform_oct_to_enface,
                              output_shape=self.enface.shape)

    @property
    def drusenfinder(self):
        """ Get and set the DrusenFinder object

        When the DrusenFinder object is set all drusen are removed.
        """
        return self._drusenfinder

    @drusenfinder.setter
    def drusenfinder(self, drusenfinder):
        self._drusen = None
        self._drusen_raw = None
        self._drusenfinder = drusenfinder

    def plot(self, ax=None, slo=True, drusen=False, bscan_region=False,
             bscan_positions=None, masks=False, region=np.s_[...], alpha=1):
        """

        Parameters
        ----------
        ax :
        slo :
        drusen :
        bscan_region :
        bscan_positions :
        masks :
        region : slice object

        Returns
        -------

        """

        if ax is None:
            ax = plt.gca()

        if slo:
            self.plot_enface(ax=ax, region=region)
        if drusen:
            self.plot_drusen(ax=ax, region=region, alpha=alpha)
        if bscan_positions is not None:
            self.plot_bscan_positions(ax=ax, bscan_positions=bscan_positions,
                                      region=region,
                                      line_kwargs={"linewidth": 0.5,
                                                   "color": "green"})
        if bscan_region:
            self.plot_bscan_region(region=region, ax=ax)

        if masks:
            self.plot_masks(region=region, ax=ax)
        # if quantification:
        #    self.plot_quantification(space=space, region=region, ax=ax,
        #    q_kwargs=q_kwargs)

    def plot_masks(self, region=np.s_[...], ax=None, color="r", linewidth=0.5):
        """

        Parameters
        ----------
        region :
        ax :
        color :
        linewidth :

        Returns
        -------

        """
        primitives = self._eyequantifier.plot_primitives(self)
        if ax is None:
            ax = plt.gca()

        for circle in primitives["circles"]:
            c = patches.Circle(circle["center"], circle["radius"],
                               facecolor='none',
                               edgecolor=color, linewidth=linewidth)
            ax.add_patch(c)

        for line in primitives["lines"]:
            x = [line["start"][0], line["end"][0]]
            y = [line["start"][1], line["end"][1]]
            ax.plot(x, y, color=color, linewidth=linewidth)

    def plot_enface(self, ax=None, region=np.s_[...]):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.enface[region], cmap="gray")

    def plot_bscan_positions(self, bscan_positions="all", ax=None,
                             region=np.s_[...], line_kwargs=None):
        if bscan_positions is None:
            bscan_positions = []
        elif bscan_positions == "all" or bscan_positions == True:
            bscan_positions = range(0, len(self))

        if line_kwargs is None:
            line_kwargs = config.line_kwargs
        else:
            line_kwargs = {**config.line_kwargs, **line_kwargs}

        for i in bscan_positions:
            bscan = self[i]
            x = np.array([bscan.StartX, bscan.EndX]) / self.ScaleXSlo
            y = np.array([bscan.StartY, bscan.EndY]) / self.ScaleYSlo

            ax.plot(x, y, **line_kwargs)

    def plot_bscan_region(self, region=np.s_[...], ax=None):

        if ax is None:
            ax = plt.gca()

        up_right_corner = (self[-1].EndX / self.ScaleXSlo,
                           self[-1].EndY / self.ScaleYSlo)
        width = (self[0].StartX - self[0].EndX) / self.ScaleXSlo
        height = (self[0].StartY - self[-1].EndY) / self.ScaleYSlo
        # Create a Rectangle patch
        rect = patches.Rectangle(up_right_corner, width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    def plot_drusen(self, ax=None, region=np.s_[...], cmap="Reds",
                    vmin=None, vmax=None, cbar=True, alpha=1):
        drusen = self.drusen_enface

        if ax is None:
            ax = plt.gca()

        if vmax is None:
            vmax = drusen.max()
        if vmin is None:
            vmin = 1

        visible = np.zeros(drusen[region].shape)
        visible[np.logical_and(vmin < drusen[region],
                               drusen[region] < vmax)] = 1

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmap), cax=cax)

        ax.imshow(drusen[region], alpha=visible * alpha, cmap=cmap, vmin=vmin,
                  vmax=vmax)

    def plot_enface_bscan(self, ax=None, n_bscan=0):
        """ Plot Slo with one selected B-Scan """
        raise NotImplementedError()

    def plot_bscans(self, bs_range=range(0, 8), cols=4, layers=None,
                    layers_kwargs=None):
        """ Plot a grid with B-Scans """
        rows = int(np.ceil(len(bs_range) / cols))
        if layers is None:
            layers = []

        fig, axes = plt.subplots(
            cols, rows, figsize=(rows * 4, cols * 4))

        with np.errstate(invalid='ignore'):
            for i in bs_range:
                bscan = self[i]
                ax = axes.flatten()[i]
                bscan.plot(ax=ax, layers=layers, layers_kwargs=layers_kwargs)


class Bscan:

    @abstractmethod
    def __init__(self, index, oct_volume):
        self._index = index
        self._oct_volume = oct_volume
        self._layer_indices = None
        self._drusen = None
        self._drusen_raw = None

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def scan(self):
        """ An array holding a single B-Scan with the commonly used contrast

        The array is of dtype <ubyte> and encodes the intensities as values
        between 0 and 255.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def scan_raw(self):
        """ An array holding a single raw B-Scan

        The dtype is not changed after the importIf available this is the
        unprocessed output of the OCT device. In any case this is the
        unprocessed data imported by eyepy.
        """
        raise NotImplementedError()

    @property
    def layer_indices(self):
        if self._layer_indices is None:
            
            self._layer_indices = {}
            for key, layer_height in self.layers.items():
                nan_indices = np.isnan(layer_height)
                col_indices = np.arange(self.shape[1])[~nan_indices]
                row_indices = np.rint(layer_height).astype(int)[~nan_indices]
                self._layer_indices[key] = (row_indices, col_indices)
                
        return self._layer_indices

    @property
    @abstractmethod
    def layers(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def layers_raw(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def drusen_raw(self):
        """ Return drusen computed from the RPE and BM layer segmentation

        The raw drusen are computed based on single B-Scans
        """
        return self._oct_volume.drusen_raw[..., self._index]

    @property
    @abstractmethod
    def drusen(self):
        """ Return filtered drusen

        Drusen are filtered based on the complete volume
        """
        return self._oct_volume.drusen[..., self._index]

    def plot(self, ax=None, layers=None, drusen=False, layers_kwargs=None,
             layers_color=None, annotation_only=False, region=np.s_[...]):
        """ Plot B-Scan with segmented Layers """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()

        if layers is None:
            layers = []
        elif layers == "all":
            layers = config.SEG_MAPPING.keys()

        if layers_kwargs is None:
            layers_kwargs = config.layers_kwargs
        else:
            layers_kwargs = {**config.layers_kwargs, **layers_kwargs}

        if layers_color is None:
            layers_color = config.layers_color
        else:
            layers_color = {**config.layers_color, **layers_color}

        if not annotation_only:
            ax.imshow(self.scan[region], cmap="gray")
        if drusen:
            visible = np.zeros(self.drusen.shape)
            visible[self.drusen] = 1.0
            ax.imshow(self.drusen, alpha=visible, cmap="Reds")
        for layer in layers:
            color = layers_color[layer]
            try:
                segmentation = self.layers[layer]
                ax.plot(segmentation, color=color, label=layer,
                        **layers_kwargs)
            except KeyError:
                warnings.warn(f"Layer '{layer}' has no Segmentation",
                              UserWarning)
