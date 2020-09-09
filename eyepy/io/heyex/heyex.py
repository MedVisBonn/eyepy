import mmap
from pathlib import Path, PosixPath
from typing import Union, IO

import numpy as np

from eyepy.core.config import SEG_MAPPING
from eyepy.core.octbase import Oct
from eyepy.io.heyex import he_vol, he_xml
from eyepy.io.utils import _get_meta_attr


class HeyexOct(Oct):
    """
    The HeyexOct object lazy loads the .vol file. It will only read exactly what
    you ask for. This means that no B-Scan image is read from the file if you
    only want the SLO or a specific field from the .vol header.
    In comparison to reading the complete file, this makes all operations on
    .vol files faster which do not require the complete file e.g. peaking at
    header fields, plotting the SLO, plotting individual B-Scans.

    .vol header
    -----------
    All fields from the .vol header can be accessed as attributes of the
    HeyexOct object.

    SLO
    ---
    The attribute `slo` of the HeyexOct object gives access to the IR SLO image
    and returns it as a numpy.ndarray of dtype `uint8`.

    B-Scans
    -------
    Individual B-Scans can be accessed using `oct_scan[index]`. The returned
    HeyexBscan object exposes all B-Scan header fields as attributes and the
    raw B-Scan image as `numpy.ndarray` of type `float32` under the attribute
    `scan_raw`. A transformed version of the raw B-Scan which is more similar to
    the Heyex experience can be accessed with the attribute `scan` and returns
    the 4th root of the raw B-Scan scaled to [0,255] as `uint8`.

    Segmentations
    -------------
    B-Scan segmentations can be accessed for individual B-Scans like
    `bscan.segmentation`. This return a numpy.ndarray of shape (NumSeg, SizeX)
    The `segmentation` attribute of the HeyexOct object returns a dictionary,
    where the key is a number and the value is a numpy.ndarray of shape
    (NumBScans, SizeX).

    """

    def __new__(cls, bscans, slo, meta, *args, **kwargs):
        for meta_attr in meta._meta_fields:
            setattr(cls, meta_attr, _get_meta_attr(meta_attr))
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, bscans, slo, meta):
        """

        Parameters
        ----------
        bscans :
        slo :
        meta :
        """
        super().__init__()
        self._bscans = bscans
        self._sloreader = slo
        self._meta = meta

        self._slo = None
        self._segmentation_raw = None

    def __getitem__(self, key):
        return self._bscans[key]

    def __len__(self):
        return self.NumBScans

    @property
    def slo(self):
        if self._slo is None:
            self._slo = self._sloreader.data
        return self._slo

    @property
    def meta(self):
        return self._meta

    @property
    def segmentation_raw(self):
        """ The segmentation as it is encoded in the .vol file.

        Segmentations for all B-Scans are  stacked such that we get a volume
        S x B x W where S are different Segmentations, B are the B-Scans and
        W is the Width of the B-Scans.
        """
        if self._segmentation_raw is None:
            self._segmentation_raw = np.stack([x.segmentation_raw
                                               for x in self._bscans], axis=1)
        return self._segmentation_raw

    @property
    def segmentation(self):
        """ All segmentations as a dictionary where the key is the layer name
        and the value is a 2D array holding the respectives layers segmentation
        height for all B-Scans of the volume. """
        nans = np.isnan(self.segmentation_raw)
        empty = np.nonzero(np.logical_or(
            np.less(self.segmentation_raw, 0, where=~nans),
            np.greater(self.segmentation_raw, self.meta.SizeZ, where=~nans)))

        data = self.segmentation_raw.copy()
        data[empty] = np.nan
        return {name: data[i, ...] for name, i in SEG_MAPPING.items()
                if np.nansum(data[i, ...]) != 0}

    @property
    def volume_raw(self):
        return np.stack([x.scan_raw for x in self._bscans], axis=-1)

    @property
    def volume(self):
        return np.stack([x.scan for x in self._bscans], axis=-1)

    @classmethod
    def read_vol(cls, file_obj):
        meta = he_vol.get_octmeta(file_obj)
        bscans = he_vol.get_bscans(file_obj, meta)
        slo = he_vol.get_slo(file_obj, meta)
        return cls(bscans, slo, meta)

    @classmethod
    def read_xml(cls, filepath):
        meta = he_xml.get_octmeta(filepath)
        bscans = he_xml.get_bscans(filepath)
        slo = he_xml.get_slo(filepath)
        return cls(bscans, slo, meta)

    @classmethod
    def from_images(cls, images):
        pass


def read_vol(file_obj: Union[str, Path, IO]):
    """ Return a HeyexOct object for a .vol file at the given file_obj

    Parameters
    ----------
    file_obj : Path to the .vol file

    Returns
    -------
    eyepy.io.heyex.HeyexOct
    """

    if type(file_obj) is str or type(file_obj) is PosixPath:
        with open(file_obj, "rb") as myfile:
            mm = mmap.mmap(myfile.fileno(), 0, access=mmap.ACCESS_READ)
            return HeyexOct.read_vol(mm)
    else:
        mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        return HeyexOct.read_vol(mm)


def read_xml(filepath: Union[str, Path]):
    """ Return a HeyexOct object from a Heyex XML export by providing the path
    to the XML

    Parameters
    ----------
    file_obj : Path to the .vol file

    Returns
    -------
    HeyexOct
    """
    return HeyexOct.read_xml(filepath)
