# -*- coding: utf-8 -*-
from eyepy.io.utils import (
    _get_date_from_xml,
    _get_datetime_from_xml,
    _get_first_as_float,
    _get_first_as_int,
    _get_first_as_str,
)


def oct_base_spec():
    return {
        # Version
        "Version": ("./SWVersion/Version", _get_first_as_str),
        # Number of A-Scans in each B-Scan, i.e. B-Scan width in pixel
        "SizeX": (
            ".//ImageType[Type='OCT']../" "OphthalmicAcquisitionContext/Width",
            _get_first_as_int,
        ),
        # Number of B-Scans in OCT scan
        "NumBScans": (".//ImageType[Type='OCT']..", len),
        # Number of samples in an A-Scan, i.e. B-Scan height in pixel
        "SizeY": (
            ".//ImageType[Type='OCT']../" "OphthalmicAcquisitionContext/Height",
            _get_first_as_int,
        ),
        # Width of a B-Scan pixel in mm
        "ScaleX": (
            ".//ImageType[Type='OCT']../" "OphthalmicAcquisitionContext/ScaleX",
            _get_first_as_float,
        ),
        # Distance between two adjacent B-Scans in mm (Not given in the XMl we
        # could try to estimate it from the B-Scan positions)
        "Distance": ("", lambda x: None),
        # Height of a B-Scan pixel in mm (In the XML the scale is given for
        # every B-Scan)
        "ScaleY": (
            ".//ImageType[Type='OCT']../" "OphthalmicAcquisitionContext/ScaleY",
            _get_first_as_float,
        ),
        # Width of the SLO image in pixel
        "SizeXSlo": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/Width",
            _get_first_as_int,
        ),
        # Height of the SLO image in pixel
        "SizeYSlo": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/Height",
            _get_first_as_int,
        ),
        # Width of a pixel in the SLO image in mm
        "ScaleXSlo": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/ScaleX",
            _get_first_as_float,
        ),
        # Height of a pixel in the SLO image in mm
        "ScaleYSlo": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/ScaleY",
            _get_first_as_float,
        ),
        # Horizontal field size of the SLO image in dgr
        "FieldSizeSlo": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/Angle",
            _get_first_as_int,
        ),
        # Scan focus in dpt
        "ScanFocus": (
            ".//ImageType[Type='LOCALIZER']../" "OphthalmicAcquisitionContext/Focus",
            _get_first_as_float,
        ),
        # Examined eye (zero terminated string). "OS": Left eye; "OD": Right eye
        "ScanPosition": (
            "./Patient/Study/Series/Laterality",
            lambda x: "OD" if x[0].text == "R" else "OS",
        ),
        # Examination time
        "ExamTime": ("./Patient/Study/Series", _get_datetime_from_xml),
        # Scan pattern type: (Not given in XML)
        #   0 = Unknown pattern,
        #   1 = Single line scan (one B-Scan only),
        #   2 = Circular scan (one B-Scan only),
        #   3 = Volume scan in ART mode,
        #   4 = Fast volume scan,
        #   5 = Radial scan (aka. star pattern)
        "ScanPattern": ("", lambda x: 0),
        # Unique identifier of this OCT-scan (zero terminated string). This is
        # identical to the number <SerID> that is part of the file name.
        # Format: n[.m] n and m are numbers. The extension .m exists only for
        # ScanPattern 1 and 2. Examples: 2390, 3433.2
        "ID": ("./Patient/Study/Series/ID", _get_first_as_int),
        # Unique identifier of the reference OCT-scan (zero terminated string).
        # Format: see ID, This ID is only present if the OCT-scan is part of a
        # progression otherwise this string is empty. For the reference scan of
        # a progression ID and ReferenceID are identical.
        "ReferenceID": ("./Patient/Study/Series/ReferenceSeries/ID", _get_first_as_int),
        # Internal patient ID used by HEYEX.
        "PID": ("./Patient/ID", _get_first_as_int),
        # User-defined patient ID (zero terminated string).
        "PatientID": ("./Patient/PatientID", _get_first_as_str),
        # Patient's date of birth
        "DOB": ("./Patient/Birthdate/Date", _get_date_from_xml),
        # Internal visit ID used by HEYEX.
        "VID": ("", lambda x: None),
        # User-defined visit ID (zero terminated string). This ID can be defined
        # in the Comment-field of the Diagnosis-tab of the Examination Data
        # dialog box. The VisitID must be defined in the first row of the
        # comment field. It has to begin with an "#" and ends with any
        # white-space character. It can contain up to 23 alpha-numeric
        # characters (excluding the "#")
        "VisitID": ("", lambda x: None),
        # Date the visit took place. Identical to the date of an examination tab
        # in HEYEX.
        "VisitDate": ("./Patient/Study/StudyDate/Date", _get_date_from_xml),
        # Type of grid used to derive thickness data. 0 No thickness data
        # available, >0 Type of grid used to derive thickness values. Thickness
        # data is only available for ScanPattern 3 and 4.
        "GridType": ("./Patient/Study/Series/ThicknessGrid/Type", _get_first_as_int),
        # Internal progression ID (zero terminated string). All scans of the
        # same progression share this ID.
        "ProgID": ("./Patient/Study/Series/ProgID", _get_first_as_str),
    }


def bscan_base_spec():
    return {
        # The same as the XML Version
        "Version": ("./SWVersion/Version", _get_first_as_str),
        # X-Coordinate of the B-Scan's start point in mm.
        "StartX": ("./OphthalmicAcquisitionContext/Start/Coord/X", _get_first_as_float),
        # Y-Coordinate of the B-Scan's start point in mm.
        "StartY": ("./OphthalmicAcquisitionContext/Start/Coord/Y", _get_first_as_float),
        # X-Coordinate of the B-Scan's end point in mm. For circle scans, this
        # is the X-Coordinate of the circle's center point.
        "EndX": ("./OphthalmicAcquisitionContext/End/Coord/X", _get_first_as_float),
        # Y-Coordinate of the B-Scan's end point in mm. For circle scans, this
        # is the Y-Coordinate of the circle's center point.
        "EndY": ("./OphthalmicAcquisitionContext/End/Coord/Y", _get_first_as_float),
        # Number of segmentation vectors
        "NumSeg": ("./Segmentation/NumSegmentations", _get_first_as_int),
        # Image quality measure. If this value does not exist, its value is set
        # to INVALID.
        "Quality": ("./OphthalmicAcquisitionContext/ImageQuality", _get_first_as_float),
        # Horizontal shift (in # of A-Scans) of the classification band against
        # the segmentation lines (for circular scan only).
        "Shift": ("", lambda x: None),
        # Intra volume transformation matrix. The values are only available for
        # volume and radial scans and if alignment is turned off, otherwise the
        # values are initialized to 0.
        "IVTrafo": ("", lambda x: None),
    }
