from __future__ import annotations

import datetime
from decimal import Decimal
import logging
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
from skimage.util import img_as_ubyte

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.eyevolume import EyeVolume
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import _get_date_from_xml
from eyepy.io.utils import _get_first_as_float
from eyepy.io.utils import _get_first_as_int
from eyepy.io.utils import _get_first_as_str
from eyepy.io.utils import get_bscan_spacing

logger = logging.getLogger(__name__)


def _optional_date(elements):
    return _get_date_from_xml(elements) if elements else None


def _image_name(reference):
    if reference is None:
        raise ValueError('Empty XML image reference')
    name = reference.replace('\\', '/').rsplit('/', 1)[-1]
    if name in {'', '.', '..'}:
        raise ValueError('Invalid XML image filename')
    return name


def _acquisition_clock(image):
    """Read a clock and UTC offset without borrowing the reference's date."""
    node = image.find('AcquisitionTime/Time')
    if node is None:
        return None
    try:
        seconds = Decimal(node.findtext('Second'))
        if not seconds.is_finite() or not 0 <= seconds < 60:
            raise ValueError('Invalid XML acquisition seconds')
        whole = int(seconds)
        fraction = int((seconds - whole) * 1000000)
        bias = node.findtext('UTCBias')
        timezone = (datetime.timezone(datetime.timedelta(minutes=int(bias)))
                    if bias is not None else None)
        return datetime.time(int(node.findtext('Hour')), int(node.findtext('Minute')),
                             whole, fraction, tzinfo=timezone).isoformat()
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ValueError('Invalid XML acquisition clock') from error


def _acquisition_fields(image, tags):
    context = image.find('OphthalmicAcquisitionContext')
    return {tag: context.findtext(tag) for tag in tags if context is not None and context.find(tag) is not None}


def _get_bool_from_xml_text(element: ElementTree.Element | None) -> bool | None:
    if element is None or element.text is None:
        return None
    return element.text.strip().lower() == 'true'


xml_format = {
    # Version
    'version': ('./SWVersion/Version', _get_first_as_str),
    # Number of A-Scans in each B-Scan, i.e. B-Scan width in pixel
    'size_x': (
        ".//ImageType[Type='OCT']../"
        'OphthalmicAcquisitionContext/Width',
        _get_first_as_int,
    ),
    # Number of B-Scans in OCT scan
    'n_bscans': (".//ImageType[Type='OCT']..", len),
    # Number of samples in an A-Scan, i.e. B-Scan height in pixel
    'size_y': (
        ".//ImageType[Type='OCT']../"
        'OphthalmicAcquisitionContext/Height',
        _get_first_as_int,
    ),
    # Width of a B-Scan pixel in mm
    'scale_x': (
        ".//ImageType[Type='OCT']../"
        'OphthalmicAcquisitionContext/ScaleX',
        _get_first_as_float,
    ),
    # Distance between two adjacent B-Scans in mm (Not given in the XMl we
    # could try to estimate it from the B-Scan positions)
    'distance': ('', lambda x: None),
    # Height of a B-Scan pixel in mm (In the XML the scale is given for
    # every B-Scan)
    'scale_y': (
        ".//ImageType[Type='OCT']../"
        'OphthalmicAcquisitionContext/ScaleY',
        _get_first_as_float,
    ),
    # Width of the SLO image in pixel
    'size_x_slo': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/Width',
        _get_first_as_int,
    ),
    # Height of the SLO image in pixel
    'size_y_slo': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/Height',
        _get_first_as_int,
    ),
    # Width of a pixel in the SLO image in mm
    'scale_x_slo': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/ScaleX',
        _get_first_as_float,
    ),
    # Height of a pixel in the SLO image in mm
    'scale_y_slo': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/ScaleY',
        _get_first_as_float,
    ),
    # Horizontal field size of the SLO image in dgr
    'field_size_slo': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/Angle',
        _get_first_as_int,
    ),
    # Scan focus in dpt
    'scan_focus': (
        ".//ImageType[Type='LOCALIZER']../"
        'OphthalmicAcquisitionContext/Focus',
        _get_first_as_float,
    ),
    # Examined eye (zero terminated string). "OS": Left eye; "OD": Right eye
    'scan_position': (
        './Patient/Study/Series/Laterality',
        lambda x: 'OD' if x[0].text == 'R' else 'OS',
    ),
    # Examination time
    'exam_time': ('./Patient/Study/Series/ExamDate/Date', _optional_date),
    'reference_exam_date': ('./Patient/Study/Series/ReferenceSeries/ExamDate/Date', _optional_date),
    # Scan pattern type: (Not given in XML)
    #   0 = Unknown pattern,
    #   1 = Single line scan (one B-Scan only),
    #   2 = Circular scan (one B-Scan only),
    #   3 = Volume scan in ART mode,
    #   4 = Fast volume scan,
    #   5 = Radial scan (aka. star pattern)
    'scan_pattern': ('', lambda x: 0),
    # Unique identifier of this OCT-scan (zero terminated string). This is
    # identical to the number <SerID> that is part of the file name.
    # Format: n[.m] n and m are numbers. The extension .m exists only for
    # ScanPattern 1 and 2. Examples: 2390, 3433.2
    'id': ('./Patient/Study/Series/ID', _get_first_as_int),
    # Unique identifier of the reference OCT-scan (zero terminated string).
    # Format: see ID, This ID is only present if the OCT-scan is part of a
    # progression otherwise this string is empty. For the reference scan of
    # a progression ID and ReferenceID are identical.
    'reference_id':
    ('./Patient/Study/Series/ReferenceSeries/ID', _get_first_as_int),
    # Internal patient ID used by HEYEX.
    'pid': ('./Patient/ID', _get_first_as_int),
    # User-defined patient ID (zero terminated string).
    'patient_id': ('./Patient/PatientID', _get_first_as_str),
    # First names of the patient
    'firstnames': ('./Patient/FirstNames', _get_first_as_str),
    # Last name of the patient
    'lastname': ('./Patient/LastName', _get_first_as_str),
    # Patient's date of birth
    'dob': ('./Patient/Birthdate/Date', _optional_date),
    'sex': ('./Patient/Sex', _get_first_as_str),
    # Internal visit ID used by HEYEX.
    'vid': ('./Patient/Study/ID', _get_first_as_int),
    # User-defined visit ID (zero terminated string). This ID can be defined
    # in the Comment-field of the Diagnosis-tab of the Examination Data
    # dialog box. The VisitID must be defined in the first row of the
    # comment field. It has to begin with an "#" and ends with any
    # white-space character. It can contain up to 23 alpha-numeric
    # characters (excluding the "#")
    'visit_id': ('', lambda x: None),
    # Date the visit took place. Identical to the date of an examination tab
    # in HEYEX.
    'visit_date': ('./Patient/Study/StudyDate/Date', _optional_date),
    # Type of grid used to derive thickness data. 0 No thickness data
    # available, >0 Type of grid used to derive thickness values. Thickness
    # data is only available for ScanPattern 3 and 4.
    'grid_type':
    ('./Patient/Study/Series/ThicknessGrid/Type', _get_first_as_int),
    # Internal progression ID (zero terminated string). All scans of the
    # same progression share this ID.
    'prog_id': ('./Patient/Study/Series/ProgID', _get_first_as_str),
}

xml_bscan_format = {
    # The same as the XML Version
    'version': ('./SWVersion/Version', _get_first_as_str),
    # X-Coordinate of the B-Scan's start point in mm.
    'start_x':
    ('./OphthalmicAcquisitionContext/Start/Coord/X', _get_first_as_float),
    # Y-Coordinate of the B-Scan's start point in mm.
    'start_y':
    ('./OphthalmicAcquisitionContext/Start/Coord/Y', _get_first_as_float),
    # X-Coordinate of the B-Scan's end point in mm. For circle scans, this
    # is the X-Coordinate of the circle's center point.
    'end_x':
    ('./OphthalmicAcquisitionContext/End/Coord/X', _get_first_as_float),
    # Y-Coordinate of the B-Scan's end point in mm. For circle scans, this
    # is the Y-Coordinate of the circle's center point.
    'end_y':
    ('./OphthalmicAcquisitionContext/End/Coord/Y', _get_first_as_float),
    # Number of segmentation vectors
    'num_seg': ('./Segmentation/NumSegmentations', _get_first_as_int),
    # Image quality measure. If this value does not exist, its value is set
    # to INVALID.
    'quality': ('./OphthalmicAcquisitionContext/ImageQuality',
                _get_first_as_float),
    # Horizontal shift (in # of A-Scans) of the classification band against
    # the segmentation lines (for circular scan only).
    'shift': ('', lambda x: None),
    # Intra volume transformation matrix. The values are only available for
    # volume and radial scans and if alignment is turned off, otherwise the
    # values are initialized to 0.
    'iv_trafo': ('', lambda x: None),
    # Bscan filename
    'scan_name': ('./ImageData/ExamURL', lambda x: _image_name(x[0].text)),
}


def get_xml_root(filepath) -> ElementTree.Element:
    parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(
        insert_comments=True, insert_pis=True))
    return ElementTree.parse(filepath, parser=parser).getroot()


class HeXmlReader:

    def __init__(self, path):
        path = Path(path)
        if not path.suffix == '.xml':
            xmls = list(path.glob('*.xml'))
            if len(xmls) == 0:
                raise FileNotFoundError(
                    'There is no .xml file under the given filepath')
            elif len(xmls) > 1:
                raise ValueError(
                    'There is more than one .xml file in the given folder.')
            path = xmls[0]
        from eyepy.io.he._native import SourceFile

        self.path = path
        self._source = SourceFile.capture(path)
        self.xml_root: ElementTree.Element = get_xml_root(self.path)

        self._source.verify()
        self._xml_snapshot = ElementTree.tostring(self.xml_root, encoding='utf-8')
        self.body = self.xml_root.find('BODY')
        if self.body is None:
            raise ValueError('The Heidelberg XML document has no BODY element')

        self.parsed_values = {
            name: func(self.body.findall(xpath))
            for name, (xpath, func) in xml_format.items()
        }

    @property
    def bscan_meta(self) -> list[EyeBscanMeta]:
        bscan_meta = []
        for bscan in self.body.findall(".//ImageType[Type='OCT'].."):
            parsed_values = {
                name: func(bscan.findall(xpath))
                for name, (xpath, func) in xml_bscan_format.items()
            }
            bscan_meta.append(
                EyeBscanMeta(
                    quality=parsed_values['quality'],
                    quality_ui=parsed_values['quality'],
                    art_mode=_get_first_as_int(bscan.findall('./OphthalmicAcquisitionContext/NumAve')),
                    heyex_xml_acquisition_time=_acquisition_clock(bscan),
                    heyex_xml_acquisition=_acquisition_fields(
                        bscan, ('Resolution', 'PositionWithinTolerance', 'EDI', 'EVI')),
                    start_pos=(parsed_values['start_x'],
                               parsed_values['start_y']),
                    end_pos=(parsed_values['end_x'], parsed_values['end_y']),
                    pos_unit='mm',
                    scan_name=parsed_values['scan_name'],
                ))

        return bscan_meta

    @property
    def meta(self) -> EyeVolumeMeta:
        bscan_meta = self.bscan_meta
        return EyeVolumeMeta(
            scale_x=self.parsed_values['scale_x'],
            scale_y=self.parsed_values['scale_y'],
            scale_z=get_bscan_spacing(bscan_meta),
            scale_unit='mm',
            laterality=self.parsed_values['scan_position'],
            visit_date=self.parsed_values['visit_date'],
            exam_time=self.parsed_values['exam_time'],
            patient=self.patient,
            id=str(self.parsed_values['id']) if self.parsed_values['id'] is not None else '',
            reference_id=str(self.parsed_values['reference_id']) if self.parsed_values['reference_id'] is not None else '',
            reference_exam_date=self.parsed_values['reference_exam_date'],
            vid=self.parsed_values['vid'],
            prog_id=self.parsed_values['prog_id'] or '',
            bscan_meta=bscan_meta,
            intensity_transform='default',
            intensity_encoding='display',
        )

    @property
    def patient(self):
        return dict(firstname=self.parsed_values['firstnames'],
                    lastname=self.parsed_values['lastname'],
                    dob=self.parsed_values['dob'],
                    pid=self.parsed_values['pid'],
                    patient_id=self.parsed_values['patient_id'],
                    sex=self.parsed_values['sex'])

    @property
    def localizer_meta(self) -> EyeEnfaceMeta:
        image = self.body.find(".//ImageType[Type='LOCALIZER']..")
        return EyeEnfaceMeta(
            scale_x=self.parsed_values['scale_x_slo'],
            scale_y=self.parsed_values['scale_y_slo'],
            scale_unit='mm',
            modality='NIR',
            laterality=self.parsed_values['scan_position'],
            field_size=self.parsed_values['field_size_slo'],
            scan_focus=self.parsed_values['scan_focus'],
            visit_date=self.parsed_values['visit_date'],
            exam_time=self.parsed_values['exam_time'],
            heyex_xml_acquisition_time=_acquisition_clock(image),
            heyex_xml_acquisition=_acquisition_fields(
                image, ('Resolution', 'AutoSensorGain', 'SensorGain', 'FixationTarget')),
            num_averaged=_get_first_as_int(image.findall('./OphthalmicAcquisitionContext/NumAve')),
        )

    @property
    def localizer(self) -> EyeEnface:
        localizer_pattern = ".//ImageType[Type='LOCALIZER']../ImageData/ExamURL"
        localizer_name = _image_name(self.body.findtext(localizer_pattern))
        import imageio.v3 as iio
        localizer = iio.imread(self.path.parent / localizer_name)
        if localizer.ndim == 3:
            localizer = img_as_ubyte(localizer[..., 0])
        else:
            localizer = img_as_ubyte(localizer)

        return EyeEnface(localizer, self.localizer_meta)

    @property
    def volume(self) -> EyeVolume:
        ## Check if scan pattern is supported by EyeVolume
        if self.parsed_values['scan_pattern'] == 2:
            msg = f'The EyeVolume object does not support scan pattern 2 (one Circular B-scan).'
            raise ValueError(msg)
        elif self.parsed_values['scan_pattern'] == 5:
            msg = f'The EyeVolume object does not support scan pattern 5 (Radial scan - star pattern).'
            raise ValueError(msg)

        bscans = []
        layer_heights = {}
        layer_manual = {}
        layer_enabled = {}
        for index, bscan_root in enumerate(
                self.body.findall(".//ImageType[Type='OCT']..")):
            scan_name = _image_name(bscan_root.findtext('./ImageData/ExamURL'))
            import imageio.v3 as iio
            img = iio.imread(self.path.parent / scan_name)
            if img.ndim == 3:
                img = img_as_ubyte(img[..., 0])
            else:
                img = img_as_ubyte(img)
            bscans.append(img)

            seglines = bscan_root.findall('.//SegLine')
            for segline in seglines:
                name = segline.find('./Name').text
                data = np.array(segline.find('./Array').text.split()).astype(
                    np.float32)
                data[data == 3.0e+38] = np.nan
                if name not in layer_heights:
                    layer_heights[name] = []
                    layer_manual[name] = []
                    layer_enabled[name] = []
                layer_heights[name].append((index, data))
                layer_manual[name].append(
                    (index, _get_bool_from_xml_text(segline.find('./Manual'))))
                layer_enabled[name].append(
                    (index, _get_bool_from_xml_text(segline.find('./Enabled'))))

        data = np.stack(bscans, axis=0)

        layer_height_maps = {
            name: np.full((data.shape[0], data.shape[2]),
                          np.nan,
                          dtype=np.float32)
            for name in layer_heights
        }
        layer_manual_maps = {
            name: [None] * data.shape[0]
            for name in layer_manual
        }
        layer_enabled_maps = {name: [None] * data.shape[0] for name in layer_enabled}

        for name, heights in layer_heights.items():
            for index, layer_height in heights:
                layer_height_maps[name][index, :] = layer_height

        for name, manual_values in layer_manual.items():
            for index, manual in manual_values:
                layer_manual_maps[name][index] = manual
        for name, enabled_values in layer_enabled.items():
            for index, enabled in enabled_values:
                layer_enabled_maps[name][index] = enabled

        localizer = self.localizer
        volume_meta = self.meta
        transform = _compute_localizer_oct_transform(volume_meta,
                                                     localizer.meta,
                                                     data.shape)
        volume = EyeVolume(
            data=data,
            meta=volume_meta,
            localizer=localizer,
            transformation=transform,
        )

        for name, height_map in layer_height_maps.items():
            volume.add_layer_annotation(
                height_map,
                name=name,
                manual=layer_manual_maps[name],
                heyex_xml_enabled=layer_enabled_maps[name],
            )

        return volume
