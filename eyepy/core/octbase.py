import numpy as np

import imageio
from eyepy.core.base import OctBase, BscanBase, OctMetaBase, BscansBase, \
    BscanMetaBase
from eyepy.core.config import SEG_MAPPING


class Oct(OctBase):

    @classmethod
    def from_images(cls, filepath):
        meta = OctMeta(filepath)
        bscans = Bscans(filepath)
        slo = None
        return cls(bscans, slo, meta)


class OctMeta(OctMetaBase):
    @staticmethod
    def _get_meta_attributes(filepath, version, **kwargs):
        pass


class Bscans(BscansBase):
    def _get_bscan(self, index):
        pass


class Bscan(BscanBase):

    @property
    def shape(self):
        pass

    @property
    def scan(self):
        pass

    @property
    def scan_raw(self):
        pass

    @property
    def layers(self):
        pass

    @property
    def layers_raw(self):
        pass


class Bscan(BscanBase):

    def __init__(self, filepath, index, oct_volume):
        """

        Parameters
        ----------
        filepath :
        bscan_index :
        oct_meta :
        """
        super().__init__(filepath, index, oct_volume)
        self._scan = None
        self._segmentation_raw = None

        self._scanpath = None

    @property
    def drusen(self):
        pass

    @property
    def drusen_raw(self):
        pass

    @property
    def shape(self):
        return self._scan.shape

    @property
    def layers_raw(self):
        if self._layers_raw is None:
            self._layers_raw = np.full(shape=(17, self.shape[1]),
                                       fill_value=np.nan, dtype="float32")
        return self._layers_raw

    @property
    def layers(self):
        data = self.layers_raw.copy()
        nans = np.isnan(self.layers_raw)
        empty = np.nonzero(np.logical_or(
            np.less(self.layers_raw, 0, where=~nans),
            np.greater(self.layers_raw, self._oct_volume.meta.SizeZ,
                       where=~nans)))
        data[empty] = np.nan
        return {name: data[i] for name, i in SEG_MAPPING.items()
                if np.nansum(data[i]) != 0}

    @property
    def scan(self):
        if self._scan is None:
            self._scan = imageio.imread(self.filepath)
            # In case 3 arrays (RGB values) are stored instead of a single array
            if self._scan.ndim == 3:
                self._scan = self._scan[..., 0]
        return self._scan

    @property
    def scan_raw(self):
        raise NotImplementedError("The Heyex XML export does not contain the"
                                  " raw OCT data.")

    @property
    def size(self):
        return np.prod(self.shape)


class BscanMeta(BscanMetaBase):
    @staticmethod
    def _get_meta_attributes(filepath, bscan_index, version, **kwargs):
        pass
