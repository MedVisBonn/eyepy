import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from eyepy.core import config
from eyepy.core.drusen import DefaultDrusenFinder


class Oct(ABC):

    @abstractmethod
    def __init__(self, bscans, enfacereader, meta,
                 drusenfinder=DefaultDrusenFinder()):
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

        self._drusen = None
        self._drusen_raw = None
        self._segmentation_raw = None

        # Set the oct volume in the Bscans object to self such that every
        # loaded B-Scan can refer to the volume
        self._bscans._oct_volume = self

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

        The dtype is not changed after the importIf available this is the
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

    def plot(self, ax=None, bscan_positions=None,
             line_kwargs={"linewidth": 0.5, "color": "green"}):
        """ Plot enface with localization of corresponding B-Scans """

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()

        if bscan_positions is None:
            bscan_positions = []
        elif bscan_positions == "all":
            bscan_positions = range(0, len(self))

        if line_kwargs is None:
            line_kwargs = config.line_kwargs
        else:
            line_kwargs = {**config.line_kwargs, **line_kwargs}

        ax.imshow(self.enface, cmap="gray")

        for i in bscan_positions:
            bscan = self[i]
            x = np.array([bscan.StartX, bscan.EndX]) / self.ScaleXSlo
            y = np.array([bscan.StartY, bscan.EndY]) / self.ScaleYSlo

            ax.plot(x, y, **line_kwargs)

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
             layers_color=None,
             annotation_only=False):
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
            ax.imshow(self.scan, cmap="gray")
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
