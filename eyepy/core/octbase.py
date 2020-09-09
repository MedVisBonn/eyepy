import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from eyepy.core import config
from eyepy.core.drusen import drusen


class Oct(ABC):

    @abstractmethod
    def __init__(self):
        self._drusen = None

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def slo(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume_raw(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def segmentation(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def segmentation_raw(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplementedError()

    @property
    def drusen(self):
        if self._drusen is None:
            self._drusen = np.stack([x.drusen for x in self._bscans], axis=-1)
        return self._drusen

    def plot(self, ax=None, bscan_positions=None, line_kwargs={"linewidth": 0.5, "color": "green"}):
        """ Plot Slo with localization of corresponding B-Scans"""

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

        ax.imshow(self.slo, cmap="gray")

        for i in bscan_positions:
            bscan = self[i]
            x = np.array([bscan.StartX, bscan.EndX]) / self.ScaleXSlo
            y = np.array([bscan.StartY, bscan.EndY]) / self.ScaleYSlo

            ax.plot(x, y, **line_kwargs)

    def plot_slo_bscan(self, ax=None, n_bscan=0):
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

    @property
    @abstractmethod
    def scan(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def scan_raw(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def segmentation(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def segmentation_raw(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def drusen(self):
        """Return drusen computed from the RPE and BM layer segmentation"""
        return drusen(self.segmentation["RPE"], self.segmentation["BM"],
                      self.scan.shape)

    def plot(self, ax=None, layers=None, drusen=False, layers_kwargs=None, layers_color=None,
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
                segmentation = self.segmentation[layer]
                ax.plot(segmentation, color=color, label=layer,
                        **layers_kwargs)
            except KeyError:
                warnings.warn(f"Layer '{layer}' has no Segmentation",
                              UserWarning)
