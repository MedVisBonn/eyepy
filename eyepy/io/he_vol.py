# -*- coding: utf-8 -*-
import functools
import os
from struct import unpack

import numpy as np
from skimage import exposure

from .base import OctReader


class BscanMeta(object):
    def __init__(self, kwargs):
        self.allowed_meta_fields = {
            "Version",
            "BScanHdrSize",
            "StartX",
            "StartY",
            "EndX",
            "EndY",
            "NumSeg",
            "OffSeg",
            "Quality",
            "Shift",
            "IVTrafo",
        }

        # Initialize all allowed meta fields with None
        [setattr(self, key, None) for key in self._allowed_meta_fields]

        # Use kwargs to fill the OctMeta objects attributes
        for key, value in kwargs.items():
            if key in self._allowed_meta_fields:
                setattr(self, key, value)
            else:
                raise ValueError(
                    "{} is not a valid field in the BscanMeta object.".format(key)
                )


class OctMeta(object):
    def __init__(self, kwargs):
        self._allowed_meta_fields = {
            "Version",
            "SizeX",
            "NumBScans",
            "SizeZ",
            "ScaleX",
            "Distance",
            "ScaleZ",
            "SizeXSlo",
            "SizeYSlo",
            "ScaleXSlo",
            "ScaleYSlo",
            "FieldSizeSlo",
            "ScanFocus",
            "ScanPosition",
            "ExamTime",
            "ScanPattern",
            "BScanHdrSize",
            "ID",
            "ReferenceID",
            "PID",
            "PatientID",
            "DOB",
            "VID",
            "VisitID",
            "VisitDate",
            "GridType",
            "GridOffset",
            "GridType1",
            "GridOffset1",
            "ProgID",
        }
        # Initialize all allowed meta fields with None
        [setattr(self, key, None) for key in self._allowed_meta_fields]

        # Use kwargs to fill the OctMeta objects attributes
        for key, value in kwargs.items():
            if key in self._allowed_meta_fields:
                setattr(self, key, value)
            else:
                raise ValueError(
                    "{} is not a valid field in the OctMeta object.".format(key)
                )


class Oct(object):
    pass


class VolReader(OctReader):
    def __init__(self, filepath):
        self.header_size = 2048
        self.slo_offset = 2048

        self.file_handle = open(filepath, "rb")
        self.meta = self.read_meta()
        self.nir_size = self.meta.SizeXSlo * self.meta.SizeYSlo
        self.bscan_size = self.meta.SizeX * self.meta.SizeZ

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.file_handle.close()

    def _bscan_meta_offset(self, scan_number):
        """

        Parameters
        ----------
        scan_number

        Returns
        -------

        """
        bscan_meta_offset = (
            self.header_size
            + self.nir_size
            + scan_number * (4 * self.bscan_size)
            + (scan_number * self.meta.BScanHdrSize)
        )

        return bscan_meta_offset

    def _bscan_offset(self, scan_number):
        """

        Parameters
        ----------
        scan_number

        Returns
        -------

        """
        return self._bscan_meta_offset(scan_number) + self.meta.BScanHdrSize

    def _bscan_seglines_offset(self, scan_number):
        """

        Parameters
        ----------
        scan_number

        Returns
        -------

        """
        return self._bscan_meta_offset(scan_number) + self.meta.OffSeg

    def read(self):
        """Returns complete Oct object.

        Returns
        -------
        """
        raise NotImplementedError("This function is not yet implemented.")

    @functools.lru_cache(maxsize=1, typed=False)
    def read_meta(self):
        """Returns OctMeta object.

        The OctMeta object contains the information found in the .vol header.

        The specification for the .vol file header shown below was found in
        https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m

        {'Version','c',0}, ...              % Version identifier: HSF-OCT-xxx, xxx = version number of the file format,
                                          Current version: xxx = 103
        {'SizeX','i',12},  ...                  % Number of A-Scans in each B-Scan, i.e. the width of each B-Scan in pixel
        {'NumBScans','i',16}, ...               % Number of B-Scans in OCT scan
        {'SizeZ','i',20}, ...                   % Number of samples in an A-Scan, i.e. the Height of each B-Scan in pixel
        {'ScaleX','d',24}, ...                  % Width of a B-Scan pixel in mm
        {'Distance','d',32}, ...                % Distance between two adjacent B-Scans in mm
        {'ScaleZ','d',40}, ...                  % Height of a B-Scan pixel in mm
        {'SizeXSlo','i',48}, ...                % Width of the SLO image in pixel
        {'SizeYSlo','i',52}, ...                % Height of the SLO image in pixel
        {'ScaleXSlo','d',56}, ...               % Width of a pixel in the SLO image in mm
        {'ScaleYSlo','d',64}, ...               % Height of a pixel in the SLO image in mm
        {'FieldSizeSlo','i',72}, ...    % Horizontal field size of the SLO image in dgr
        {'ScanFocus','d',76}, ...               % Scan focus in dpt
        {'ScanPosition','c',84}, ...    % Examined eye (zero terminated string). "OS" for left eye; "OD" for right eye
        {'ExamTime','i',88}, ...                % Examination time. The structure holds an unsigned 64-bit date and time value and
                                          represents the number of 100-nanosecond units since the beginning of January 1, 1601.
        {'ScanPattern','i',96}, ...             % Scan pattern type: 0 = Unknown pattern, 1 = Single line scan (one B-Scan only),
                                          2 = Circular scan (one B-Scan only), 3 = Volume scan in ART mode,
                                          4 = Fast volume scan, 5 = Radial scan (aka. star pattern)
        {'BScanHdrSize','i',100}, ...   % Size of the Header preceding each B-Scan in bytes
        {'ID','c',104}, ...                             % Unique identifier of this OCT-scan (zero terminated string). This is identical to
                                          the number <SerID> that is part of the file name. Format: n[.m] n and m are
                                          numbers. The extension .m exists only for ScanPattern 1 and 2. Examples: 2390, 3433.2
        {'ReferenceID','c',120}, ...    % Unique identifier of the reference OCT-scan (zero terminated string). Format:
                                          see ID, This ID is only present if the OCT-scan is part of a progression otherwise
                                          this string is empty. For the reference scan of a progression ID and ReferenceID
                                          are identical.
        {'PID','i',136}, ...                    % Internal patient ID used by HEYEX.
        {'PatientID','c',140}, ...              % User-defined patient ID (zero terminated string).
        {'Padding','c',161}, ...                % To align next member to 4-byte boundary.
        {'DOB','date',164}, ...                 % Patient's date of birth
        {'VID','i',172}, ...                    % Internal visit ID used by HEYEX.
        {'VisitID','c',176}, ...                % User-defined visit ID (zero terminated string). This ID can be defined in the
                                          Comment-field of the Diagnosis-tab of the Examination Data dialog box. The VisitID
                                          must be defined in the first row of the comment field. It has to begin with an "#"
                                          and ends with any white-space character. It can contain up to 23 alpha-numeric
                                          characters (excluding the "#").
        {'VisitDate','date',200}, ...   % Date the visit took place. Identical to the date of an examination tab in HEYEX.
        {'GridType','i',208}, ...               % Type of grid used to derive thickness data. 0 No thickness data available,
                                          >0 Type of grid used to derive thickness  values. Seeter "Thickness Grid"     for more
                                          details on thickness data, Thickness data is only available for ScanPattern 3 and 4.
        {'GridOffset','i',212}, ...             % File offset of the thickness data in the file. If GridType is 0, GridOffset is 0.
        {'GridType1','i',216}, ...              % Type of a 2nd grid used to derive a 2nd set of thickness data.
        {'GridOffset1','i',220}, ...    % File offset of the 2 nd thickness data set in the file.
        {'ProgID','c',224}, ...                 % Internal progression ID (zero terminated string). All scans of the same
                                          progression share this ID.
        {'Spare','c',258}};                     % Spare bytes for future use. Initialized to 0.

        Returns
        -------
        """
        self.file_handle.seek(0)

        # Read raw hdr
        (
            Version,
            SizeX,
            NumBScans,
            SizeZ,
            ScaleX,
            Distance,
            ScaleZ,
            SizeXSlo,
            SizeYSlo,
            ScaleXSlo,
            ScaleYSlo,
            FieldSizeSlo,
            ScanFocus,
            ScanPosition,
            ExamTime,
            ScanPattern,
            BScanHdrSize,
            ID,
            ReferenceID,
            PID,
            PatientID,
            Padding,
            DOB,
            VID,
            VisitID,
            VisitDate,
            GridType,
            GridOffset,
            GridType1,
            GridOffset1,
            ProgID,
            Spare,
        ) = unpack(
            "=12siiidddiiddid4sQii16s16si21s3sdi24sdiiii34s1790s",
            self.file_handle.read(2048),
        )

        # Format hdr properly
        hdr = {
            "Version": Version.decode("ascii").replace("\x00", ""),
            "SizeX": SizeX,
            "NumBScans": NumBScans,
            "SizeZ": SizeZ,
            "ScaleX": ScaleX,
            "Distance": Distance,
            "ScaleZ": ScaleZ,
            "SizeXSlo": SizeXSlo,
            "SizeYSlo": SizeYSlo,
            "ScaleXSlo": ScaleXSlo,
            "ScaleYSlo": ScaleYSlo,
            "FieldSizeSlo": FieldSizeSlo,
            "ScanFocus": ScanFocus,
            "ScanPosition": ScanPosition.decode("ascii").replace("\x00", ""),
            "ExamTime": ExamTime,
            "ScanPattern": ScanPattern,
            "BScanHdrSize": BScanHdrSize,
            "ID": ID.decode("ascii").replace("\x00", ""),
            "ReferenceID": ReferenceID.decode("ascii").replace("\x00", ""),
            "PID": PID,
            "PatientID": PatientID.decode("ascii").replace("\x00", ""),
            "DOB": DOB,
            "VID": VID,
            "VisitID": VisitID.decode("ascii").replace("\x00", ""),
            "VisitDate": VisitDate,
            "GridType": GridType,
            "GridOffset": GridOffset,
            "GridType1": GridType1,
            "GridOffset1": GridOffset1,
            "ProgID": ProgID.decode("ascii").replace("\x00", ""),
        }

        return OctMeta(hdr)

    def read_bscan_meta(self, scan_number):
        """Returns the B-Scan meta data for a specific B-Scan.

        The specification of the B-scan header shown below was found in:
        https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m

        {'Version','c',0}, ...              % Version identifier (zero terminated string). Version Format: "HSF-BS-xxx,
                                          xxx = version number of the B-Scan header format. Current version: xxx = 103
        {'BScanHdrSize','i',12}, ...    % Size of the B-Scan header in bytes. It is identical to the same value of the
                                          file header.
        {'StartX','d',16}, ...              % X-Coordinate of the B-Scan's start point in mm.
        {'StartY','d',24}, ...              % Y-Coordinate of the B-Scan's start point in mm.
        {'EndX','d',32}, ...                    % X-Coordinate of the B-Scan's end point in mm. For circle scans, this is the
                                          X-Coordinate of the circle's center point.
        {'EndY','d',40}, ...                    % Y-Coordinate of the B-Scan's end point in mm. For circle scans, this is the
                                          Y-Coordinate of the circle's center point.
        {'NumSeg','i',48}, ...              % Number of segmentation vectors
        {'OffSeg','i',52}, ...              % Offset of the array of segmentation vectors relative to the beginning of this
                                          B-Scan header.
        {'Quality','f',56}, ...             % Image quality measure. If this value does not exist, its value is set to INVALID.
        {'Shift','i',60}, ...                   % Horizontal shift (in # of A-Scans) of the classification band against the
                                          segmentation lines (for circular scan only).
        {'IVTrafo','f',64}, ...             % Intra volume transformation matrix. The values are only available for volume and
                                          radial scans and if alignment is turned off, otherwise the values are initialized
                                          to 0.
        {'Spare','c',88}};              % Spare bytes for future use.

        Parameters
        ----------
        scan_number

        Returns
        -------
        """
        self.file_handle.seek(self._bscan_meta_offset(scan_number))

        # Read B-scan header
        header_tail_size = self.meta.BScanHdrSize - 68
        bs_header = unpack(
            "=12siddddiifif" + str(header_tail_size) + "s",
            self.file_handle.read(self.meta.BScanHdrSize),
        )
        bscan_meta = {
            "Version": bs_header[0].rstrip(),
            "BScanHdrSize": bs_header[1],
            "StartX": bs_header[2],
            "StartY": bs_header[3],
            "EndX": bs_header[4],
            "EndY": bs_header[5],
            "NumSeg": bs_header[6],
            "OffSeg": bs_header[7],
            "Quality": bs_header[8],
            "Shift": bs_header[9],
            "IVTrafo": bs_header[10],
        }

        return BscanMeta(bscan_meta)

    def read_bscan_seglines(self, scan_number):
        """

        Parameters
        ----------
        scan_number

        Returns
        -------

        """
        self.file_handle.seek(self._bscan_seglines_offset(scan_number))
        seg_size = self.meta.NumSeg * self.meta.SizeX

        seg_lines = unpack(
            "=" + str(seg_size) + "f", self.file_handle.read(seg_size * 4)
        )
        seg_lines = np.asarray(seg_lines, dtype="float32")
        seg_lines = seg_lines.reshape(self.meta.NumSeg, self.meta.SizeX)

        return seg_lines

    def read_bscan(self, scan_number):
        """

        Parameters
        ----------
        filepath
        scan_number

        Returns
        -------

        """
        self.file_handle.seek(self._bscan_offset())

        bscan_img = unpack(
            "=" + str(self.bscan_size) + "f", self.file_handle.read(self.bscan_size * 4)
        )
        bscan_img = np.asarray(bscan_img, dtype="float32")
        bscan_img[bscan_img > 1] = 0
        bscan_img = bscan_img.reshape(self.self.meta.SizeZ, self.meta.SizeX)

        return bscan_img

    def read_bscans(self):
        """Returns only the B-scans.

        Returns
        -------
        """
        bscans = np.zeros((self.meta.SizeZ, self.meta.SizeX, self.meta.NumBScans))

        for scan_number in range(self.meta.NumBScans):
            bscans[:, :, scan_number] = self.read_bscan(scan_number)

        return bscans

    def read_seglines(self):
        """

        Returns
        -------

        """
        bscan_seglines = []

        for scan_number in range(self.meta.NumBScans):
            bscan_seglines.append(self.read_bscan_seglines(scan_number))

        return bscan_seglines

    @functools.lru_cache(maxsize=4, typed=False)
    def read_nir(self):
        """Returns only the near-infrared fundus reflectance (NIR) acquired by
        an scanning laser ophthalmoscope (SLO)

        Returns
        -------
        """
        self.file_handle.seek(self.slo_offset)

        slo_img = unpack(
            "=" + str(self.nir_size) + "B", self.file_handle.read(self.nir_size)
        )
        slo_img = np.asarray(slo_img, dtype="uint8")
        slo_img = slo_img.reshape(self.meta.SizeXSlo, self.meta.SizeYSlo)

        return slo_img


def hist_match(source, template=None):
    """Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape

    # Flatten the input arrays
    source = source.ravel()
    if template is not None:
        template = template.ravel()
        t_values, t_counts = np.unique(template, return_counts=True)
        np.savez(
            "oct_refhist.npz", t_values=np.array(t_values), t_counts=np.array(t_counts)
        )
    else:
        refhist_path = os.path.join(os.path.dirname(__file__), "oct_refhist.npz")
        ref_vals = np.load(refhist_path)
        t_values, t_counts = ref_vals["t_values"], ref_vals["t_counts"]

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def improve_contrast(bscans, method="hist_match"):
    """

    Parameters
    ----------
    bscans
    method

    Returns
    -------

    """
    if method == "hist_match":
        bscans = (bscans * 255).astype("uint8")
        bscans = hist_match(bscans)
    elif method == "equalize":
        bscans = (bscans * 255).astype("uint8")
        bscans = exposure.equalize_hist(bscans)
    else:
        bscans = (bscans * 255).astype("uint8")

    return bscans
