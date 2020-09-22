import mmap
from pathlib import Path, PosixPath
from typing import Union, IO

from eyepy.core.octbase import Oct
from eyepy.io.heyex import he_vol, he_xml


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

    def __init__(self, bscans, slo, meta):
        """

        Parameters
        ----------
        bscans :
        slo :
        meta :
        """
        super().__init__(bscans, slo, meta)

    @property
    def shape(self):
        return (self.SizeZ, self.SizeX, self.NumBScans)

    @property
    def fovea_pos(self):
        return self.SizeXSlo / 2, self.SizeYSlo / 2

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
    filepath : Path to the .xml file

    Returns
    -------
    HeyexOct
    """
    return HeyexOct.read_xml(filepath)
