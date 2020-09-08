import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from eyepy.core import config


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
            self._drusen = np.stack([x.drusen for x in self._bscans], axis=1)
        return self._drusen

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

    @property
    @abstractmethod
    def drusen(self):
        """ Compute drusen from the RPE layer segmentation.

        First estimate the normal RPE by fitting a polynomial to the RPE. Then
        compute drusen as the area between the RPE and the normal RPE
        """
        rpe_height = self.segmentation["RPE"]
        bm_height = self.segmentation["BM"]
        normal_rpe_height = normal_RPE_estimation()

        # def normal_RPE_estimation(layer, degree=3, it=3, s_ratio=1, \
        #                          farDiff=5, ignoreFarPoints=True, returnImg=False, \
        #                          useBM=False, useWarping=True, xloc=[], yloc=[], polyFitType="Reguilarized"):
        """
        Given the RPE layer, estimate the normal RPE. By first warping the RPE
        layer with respect to the BM layer. Then fit a third degree polynomial
        on the RPE layer, and warp the resulting curve back.
        """
        # if (useWarping):

        #    yn, xn = warp_BM(layer)
        #    return yn, xn

        # def warp_BM(seg_img, returnWarpedImg=False):
        #    """
        #    Warp seg_img with respect to the BM layer in the segmentation image.
        #    Return the location of the warped RPE layer in the end. If
        #    returnWarpedImg is set to True, return the whole warped seg_img.
        #    """
        h, w = self.scan.shape
        # yr, xr = get_RPE_location(seg_img)
        # yb, xb = get_BM_location(seg_img)
        rmask = np.zeros((h, w), dtype='int')
        bmask = np.zeros((h, w), dtype='int')

        rmask[yr, xr] = 255
        bmask[yb, xb] = 255

        vis_img = np.copy(seg_img)

        wvector = np.empty((w), dtype='int')
        wvector.fill(h - (h / 2))

        zero_x = []
        zero_part = False
        last_nonzero_diff = 0
        for i in range(w):
            # get BM height(s) in A-Scan
            bcol = np.where(bmask[:, i] > 0)[0]
            # Set to half the image height - largest BM height in this column
            # (we have only one height per column) This is how much the BM has
            # to get shifted to align to the horizontal center line
            wvector[i] = wvector[i] - np.max(bcol) if len(bcol) > 0 else 0

            # In case there is no BM given in this column
            if (len(bcol) == 0):
                # Keep track of the gaps in the BM
                zero_part = True
                zero_x.append(i)
            if (len(bcol) > 0 and zero_part):
                # Set the gaps found sofar to the current BM diff
                diff = wvector[i]
                zero_part = False
                wvector[zero_x] = diff
                zero_x = []
            if (len(bcol) > 0):
                last_nonzero_diff = wvector[i]
            if (i == w - 1 and zero_part):
                # If we are done and there are open gaps fill it with last BM diff value
                wvector[zero_x] = last_nonzero_diff

        shifted = np.zeros(vis_img.shape)
        nrmask = np.zeros((h, w), dtype='int')
        nbmask = np.zeros((h, w), dtype='int')
        # Shift BM, RPE and and combination of both
        for i in range(w):
            nrmask[:, i] = np.roll(rmask[:, i], wvector[i])
            nbmask[:, i] = np.roll(bmask[:, i], wvector[i])
            shifted[:, i] = np.roll(vis_img[:, i], wvector[i])
        # now shift the RPE location vector as well
        shifted_yr = []
        for x, y in enumerate(rpe_height):
            shifted_yr.append(y[i] + wvector[x])

        # yn, xn = normal_RPE_estimation(rmask, it=5, useWarping=False, \
        #                               xloc=xr, yloc=shifted_yr)
        xloc = range(len(rpe_height))
        yloc = shifted_yr
        y = yloc
        x = xloc
        tmpx = np.copy(x)
        tmpy = np.copy(y)
        origy = np.copy(y)
        origx = np.copy(x)
        finalx = np.copy(tmpx)
        finaly = tmpy

        degree = 3
        it = 5
        s_ratio = 1
        farDiff = 5
        ignoreFarPoints = True
        polyFitType = "Reguilarized"

        for i in range(it):
            if (s_ratio > 1):
                s_rate = len(tmpx) / s_ratio
                rand = np.random.rand(s_rate) * len(tmpx)
                rand = rand.astype('int')

                sx = tmpx[rand]
                sy = tmpy[rand]
                if (polyFitType == 'None'):
                    z = np.polyfit(sx, sy, deg=degree)
                else:
                    z = self.compute_reguilarized_fit(sx, sy, deg=degree)
            else:
                if (polyFitType == 'None'):
                    z = np.polyfit(tmpx, tmpy, deg=degree)
                else:
                    z = self.compute_reguilarized_fit(tmpx, tmpy, deg=degree)
            p = np.poly1d(z)

            new_y = p(finalx).astype('int')
            if (ignoreFarPoints):
                tmpx = []
                tmpy = []
                for i in range(0, len(origx)):
                    diff = new_y[i] - origy[i]
                    if diff < farDiff:
                        tmpx.append(origx[i])
                        tmpy.append(origy[i])
            else:
                tmpy = np.maximum(new_y, tmpy)
            finaly = new_y

        yn, xn = finaly, finalx

        for i in range(len(xn)):
            yn[i] = yn[i] - wvector[xn[i]]

        return yn, xn

        # Create drusen map
        drusen = np.zeros(self.scan.shape)
        # Exclude normal RPE and RPE from the drusen area.
        drusen[normal_rpe_height + 1:rpe_height] = 1
        return drusen

    def compute_reguilarized_fit(self, x, y, deg):
        resMat = np.zeros((deg + 1, deg + 1))
        for d in range(deg + 1):
            z = np.polyfit(x, y, deg=d)
            for i in range(len(z)):
                resMat[d, -1 - i] = z[-1 - i]
        weightedAvg = np.average(resMat, axis=0, weights=[1., 1., 0.1 * 2, 0.1 ** 4])
        return weightedAvg

    def plot(self, ax=None, layers=None, layers_kwargs=None, layers_color=None, layers_only=False):
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

        if not layers_only:
            ax.imshow(self.scan, cmap="gray")
        for layer in layers:
            color = layers_color[layer]
            try:
                segmentation = self.segmentation[layer]
                ax.plot(segmentation, color=color, label=layer,
                        **layers_kwargs)
            except KeyError:
                warnings.warn(f"Layer '{layer}' has no Segmentation",
                              UserWarning)
