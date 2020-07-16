import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from eyepy.core import config


class Oct(ABC):

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

    def plot(self, ax=None, bscan_positions=None, line_kwargs=None):
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
            ax.plot(np.array([bscan.StartX, bscan.EndX]) / bscan.ScaleXSlo,
                    np.array([bscan.StartX, bscan.EndX]) / bscan.ScaleXSlo,
                    **line_kwargs)

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

    def plot(self, ax=None, layers=None, layers_kwargs=None, layers_color=None):
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

        ax.imshow(self.scan, cmap="gray")
        for layer in layers:
            color = layers_color[layer]
            try:
                segmentation = self.segmentation[layer]
                ax.plot(segmentation, color=color,
                        **layers_kwargs)
            except KeyError:
                warnings.warn(f"Layer '{layer}' has no Segmentation",
                              UserWarning)
