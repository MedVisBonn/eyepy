# -*- coding: utf-8 -*-
from datetime import datetime

from eyepy.io.utils import _clean_ascii, _date_from_seconds, _get_first


# "HSF-OCT-103"
def oct_base_spec():
    return {  # Version identifier: HSF-OCT-xxx
        "Version": ("12s", _clean_ascii),
        # Number of A-Scans in each B-Scan, i.e. B-Scan width in pixel
        "SizeX": ("I", _get_first),
        # Number of B-Scans in OCT scan
        "NumBScans": ("I", _get_first),
        # Number of samples in an A-Scan, i.e. B-Scan height in pixel
        "SizeY": ("I", _get_first),
        # Width of a B-Scan pixel in mm
        "ScaleX": ("d", _get_first),
        # Distance between two adjacent B-Scans in mm
        "Distance": ("d", _get_first),
        # Height of a B-Scan pixel in mm
        "ScaleY": ("d", _get_first),
        # Width of the SLO image in pixel
        "SizeXSlo": ("I", _get_first),
        # Height of the SLO image in pixel
        "SizeYSlo": ("I", _get_first),
        # Width of a pixel in the SLO image in mm
        "ScaleXSlo": ("d", _get_first),
        # Height of a pixel in the SLO image in mm
        "ScaleYSlo": ("d", _get_first),
        # Horizontal field size of the SLO image in dgr
        "FieldSizeSlo": ("I", _get_first),
        # Scan focus in dpt
        "ScanFocus": ("d", _get_first),
        # Examined eye (zero terminated string). "OS": Left eye; "OD": Right eye
        "ScanPosition": ("4s", _clean_ascii),
        # Examination time. The structure holds an unsigned 64-bit date and time
        # value. It is encoded as 100ns units since beginning of January 1, 1601
        "ExamTime": (
            "Q",
            lambda x: _date_from_seconds(x[0], datetime(1601, 1, 1), 1e-7),
        ),
        # Scan pattern type:
        #   0 = Unknown pattern,
        #   1 = Single line scan (one B-Scan only),
        #   2 = Circular scan (one B-Scan only),
        #   3 = Volume scan in ART mode,
        #   4 = Fast volume scan,
        #   5 = Radial scan (aka. star pattern)
        "ScanPattern": ("i", _get_first),
        # Size of the Header preceding each B-Scan in bytes
        "BScanHdrSize": ("I", _get_first),
        # Unique identifier of this OCT-scan (zero terminated string). This is
        # identical to the number <SerID> that is part of the file name.
        # Format: n[.m] n and m are numbers. The extension .m exists only for
        # ScanPattern 1 and 2. Examples: 2390, 3433.2
        "ID": ("16s", _clean_ascii),
        # Unique identifier of the reference OCT-scan (zero terminated string).
        # Format: see ID, This ID is only present if the OCT-scan is part of a
        # progression otherwise this string is empty. For the reference scan of
        # a progression ID and ReferenceID are identical.
        "ReferenceID": ("16s", _clean_ascii),
        # Internal patient ID used by HEYEX.
        "PID": ("I", _get_first),
        # User-defined patient ID (zero terminated string).
        "PatientID": ("24s", _clean_ascii),
        # Patient's date of birth
        "DOB": (
            "d",
            lambda x: _date_from_seconds(x[0], datetime(1899, 12, 30), 60 * 60 * 24),
        ),
        # Internal visit ID used by HEYEX.
        "VID": ("I", _get_first),
        # User-defined visit ID (zero terminated string). This ID can be defined
        # in the Comment-field of the Diagnosis-tab of the Examination Data
        # dialog box. The VisitID must be defined in the first row of the
        # comment field. It has to begin with an "#" and ends with any
        # white-space character. It can contain up to 23 alpha-numeric
        # characters (excluding the "#")
        "VisitID": ("24s", _clean_ascii),
        # Date the visit took place. Identical to the date of an examination tab
        # in HEYEX.
        "VisitDate": (
            "d",
            lambda x: _date_from_seconds(x[0], datetime(1899, 12, 30), 60 * 60 * 24),
        ),
        # Type of grid used to derive thickness data. 0 No thickness data
        # available, >0 Type of grid used to derive thickness values. Thickness
        # data is only available for ScanPattern 3 and 4.
        "GridType": ("I", _get_first),
        # File offset of the thickness data in the file. If GridType is 0,
        # GridOffset is 0.
        "GridOffset": ("I", _get_first),
        # Type of a 2nd grid used to derive a 2nd set of thickness data.
        "GridType1": ("I", _get_first),
        # File offset of the 2 nd thickness data set in the file.
        "GridOffset1": ("I", _get_first),
        # Internal progression ID (zero terminated string). All scans of the
        # same progression share this ID.
        "ProgID": ("34s", _clean_ascii),
        # Spare bytes for future use. Initialized to 0.
        "__empty": ("1790s", _get_first),
    }


# "HSF-BS-103"
def bscan_base_spec():
    return {  # Version identifier (zero terminated string).
        "Version": ("12s", _clean_ascii),
        # Size of the B-Scan header in bytes. It is identical to the same value
        # of the file header.
        "BScanHdrSize": ("i", _get_first),
        # X-Coordinate of the B-Scan's start point in mm.
        "StartX": ("d", _get_first),
        # Y-Coordinate of the B-Scan's start point in mm.
        "StartY": ("d", _get_first),
        # X-Coordinate of the B-Scan's end point in mm. For circle scans, this
        # is the X-Coordinate of the circle's center point.
        "EndX": ("d", _get_first),
        # Y-Coordinate of the B-Scan's end point in mm. For circle scans, this
        # is the Y-Coordinate of the circle's center point.
        "EndY": ("d", _get_first),
        # Number of segmentation vectors
        "NumSeg": ("I", _get_first),
        # Offset of the array of segmentation vectors relative to the beginning
        # of this B-Scan header.
        "OffSeg": ("I", _get_first),
        # Image quality measure. If this value does not exist, its value is set
        # to INVALID.
        "Quality": ("f", _get_first),
        # Horizontal shift (in # of A-Scans) of the classification band against
        # the segmentation lines (for circular scan only).
        "Shift": ("I", _get_first),
        # Intra volume transformation matrix. The values are only available for
        # volume and radial scans and if alignment is turned off, otherwise the
        # values are initialized to 0.
        "IVTrafo": ("ffffff", lambda x: x),
        # Spare bytes for future use.
        "__empty": ("168s", _get_first),
    }
