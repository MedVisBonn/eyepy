"""Source-derived XML bundles and explicit HEYEX export context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import datetime
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from eyepy.io.he._native import write_file

if TYPE_CHECKING:
    from eyepy.core.eyevolume import EyeVolume


@dataclass(frozen=True)
class HeXmlExportContext:
    """Receiving-database identifiers and explicitly selected export settings.

    Values are supplied by the exporting installation, not inferred from
    a paired target XML. Optional acquisition fields supplement source
    metadata. Metric calibration is taken from the volume; this context
    does not establish exact HEYEX geometry or thickness-grid
    statistics.
    """

    pid: int
    study_id: int
    series_id: int
    software_name: str
    software_version: str
    align_followup_to_reference: bool
    reference_id: int | None = None
    reference_exam_date: str | datetime.date | None = None
    visit_date: str | datetime.date | None = None
    patient_id: str | None = None
    patient_uids: tuple[tuple[str, str], ...] = ()
    study_uid: str | None = None
    series_uid: str | None = None
    reference_uid: str | None = None
    examined_structure: str | None = None
    modality_procedure: str | None = None
    series_type: str | None = None
    operator: str | None = None
    referring_physician: str | None = None
    equipment: Mapping[str, str] = field(default_factory=dict)
    oct_field_size: tuple[float, float] | None = None
    utc_bias_minutes: int | None = None
    localizer_time: str | datetime.datetime | datetime.time | None = None
    localizer_acquisition: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    oct_acquisition: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    filename: str = 'export.xml'
    image_names: tuple[str, ...] | None = None
    exam_url_prefix: str = ''
    scale_precision: int = 4
    position_precision: int = 3


def _xml_text(value):
    text = str(value)
    if any(not (char in '\t\n\r' or '\x20' <= char <= '\ud7ff' or
                '\ue000' <= char <= '\ufffd' or '\U00010000' <= char <= '\U0010ffff')
           for char in text):
        raise ValueError('Text contains characters forbidden by XML 1.0')
    return text


def serialize_heyex_xml(root: ET.Element) -> bytes:
    """Serialize the observed UTF-8, unindented, CRLF XML export profile.

    This proves no image processing or metadata derivation. Mixed XML
    content, comments and processing instructions are outside this
    generated profile; native rewriting retains those original bytes.
    """
    lines = []

    def visit(node):
        if not isinstance(node.tag, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', node.tag) is None:
            raise ValueError('Generated XML profile requires ordinary elements')
        attributes = ''
        for key, value in node.attrib.items():
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', key) is None:
                raise ValueError('Generated XML profile requires ordinary attribute names')
            value = escape(_xml_text(value), {'"': '&quot;'})
            value = value.replace('\r', '&#13;').replace('\n', '&#10;').replace('\t', '&#9;')
            attributes += f' {key}="{value}"'
        opening = f'<{node.tag}{attributes}'
        if len(node):
            if node.text and node.text.strip():
                raise ValueError('Generated XML profile does not support mixed content')
            lines.append(opening + '>')
            for child in node:
                visit(child)
                if child.tail and child.tail.strip():
                    raise ValueError('Generated XML profile does not support mixed content')
            lines.append(f'</{node.tag}>')
        elif node.text is None or node.text == '':
            lines.append(opening + '/>')
        else:
            # XML normalizes literal CR on read; character references retain
            # CR data inside a scalar without affecting document line endings.
            text = escape(_xml_text(node.text)).replace('\r', '&#13;')
            lines.append(opening + '>' + text + f'</{node.tag}>')

    visit(root)
    return ('\r\n'.join(lines) + '\r\n').encode('utf-8')


def _add(parent, tag, value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError('XML acquisition scalars must be finite')
    node = ET.SubElement(parent, tag)
    node.text = ('true' if value else 'false') if isinstance(value, (bool, np.bool_)) else _xml_text(value)
    return node


def _date(parent, tag, value):
    if value is None:
        return
    date = value.date() if isinstance(value, datetime.datetime) else value
    if isinstance(date, str):
        date = datetime.datetime.fromisoformat(date).date()
    if not isinstance(date, datetime.date):
        raise ValueError('XML date must be an ISO date or date object')
    node = ET.SubElement(ET.SubElement(parent, tag), 'Date')
    for key, part in [('Year', date.year), ('Month', date.month), ('Day', date.day)]:
        _add(node, key, f'{part:04d}' if key == 'Year' else f'{part:02d}')


def _time(parent, value, bias):
    if value is None:
        return
    if isinstance(value, str):
        value = (datetime.datetime.fromisoformat(value) if 'T' in value or ' ' in value
                 else datetime.time.fromisoformat(value))
    offset = value.utcoffset() if isinstance(value, (datetime.datetime, datetime.time)) else None
    if isinstance(value, datetime.datetime):
        value = value.timetz()
    if not isinstance(value, datetime.time):
        raise ValueError('XML acquisition time must be an ISO time or datetime')
    if offset is not None:
        seconds = offset.total_seconds()
        if seconds % 60:
            raise ValueError('XML UTC bias requires whole minutes')
        if bias is not None and bias != seconds / 60:
            raise ValueError('Explicit XML UTC bias disagrees with acquisition time')
        bias = int(seconds / 60)
    if bias is None:
        raise ValueError('Naive acquisition times require an explicit utc_bias_minutes')
    if isinstance(bias, bool) or not isinstance(bias, int) or not -1440 < bias < 1440:
        raise ValueError('Invalid XML UTC bias in minutes')
    # Observed XML exports use millisecond clocks. Round with carry rather
    # than emitting an invalid Second=60 at a minute/day boundary.
    rounded = (datetime.datetime(2000, 1, 1, value.hour, value.minute, value.second) +
               datetime.timedelta(milliseconds=round(value.microsecond / 1000)))
    node = ET.SubElement(ET.SubElement(parent, 'AcquisitionTime'), 'Time')
    _add(node, 'Hour', rounded.hour)
    _add(node, 'Minute', rounded.minute)
    _add(node, 'Second', f'{rounded.second + rounded.microsecond / 1e6:.3f}')
    _add(node, 'UTCBias', bias)


def _filename(name, suffixes):
    if (not isinstance(name, str) or not name or name in {'.', '..'} or
            any(char in name for char in '<>:"/\\|?*') or any(ord(char) < 32 for char in name) or
            Path(name).suffix.lower() not in suffixes):
        raise ValueError('Export filenames must be plain filenames with the required extension')
    return name


def _layer_flag(flags, index, count):
    if flags is None:
        return None
    if not isinstance(flags, (list, tuple, np.ndarray)) or len(flags) != count:
        raise ValueError('XML layer flags disagree with the B-scan count')
    value = flags[index]
    if value is not None and not isinstance(value, (bool, np.bool_)):
        raise ValueError('XML layer flags must be boolean or unknown')
    return value


class GeneratedXmlBundle:
    """Generate a document and stream its source-derived TIFF assets."""

    def __init__(self, volume: EyeVolume, context: HeXmlExportContext, include_segmentation: bool):
        if not isinstance(context, HeXmlExportContext):
            raise TypeError('Generating XML requires a HeXmlExportContext')
        self.volume, self.context = volume, context
        self.include_segmentation = include_segmentation
        self.filename = _filename(context.filename, {'.xml'})
        names = context.image_names
        if names is None:
            names = ('localizer.tif', *(f'bscan_{i + 1:06d}.tif' for i in range(volume.size_z)))
        if len(names) != volume.size_z + 1:
            raise ValueError('XML image_names must include one localizer and every B-scan')
        self.names = tuple(_filename(name, {'.tif', '.tiff'}) for name in names)
        if len({name.casefold() for name in (self.filename, *self.names)}) != len(self.names) + 1:
            raise ValueError('XML bundle filenames must be distinct, including case-insensitive spelling')

    def document(self):
        volume, ctx = self.volume, self.context
        for value in (ctx.pid, ctx.study_id, ctx.series_id):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('XML database IDs must be nonnegative integers')
        if ctx.reference_id is not None and (isinstance(ctx.reference_id, bool) or not isinstance(ctx.reference_id, int) or ctx.reference_id < 0):
            raise ValueError('XML database IDs must be nonnegative integers')
        if any(not isinstance(value, str) or not value for value in (ctx.software_name, ctx.software_version)):
            raise ValueError('XML export software name and version are required')
        if not isinstance(ctx.align_followup_to_reference, bool):
            raise ValueError('An explicit boolean XML alignment setting is required')
        for precision in (ctx.scale_precision, ctx.position_precision):
            if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 15:
                raise ValueError('Invalid XML numeric precision')
        if volume.scale_unit != 'mm' or volume.localizer.scale_unit != 'mm':
            raise ValueError('Generated Heidelberg XML requires metric calibration in mm')
        if volume.laterality not in {'OD', 'OS'}:
            raise ValueError('Generated Heidelberg XML requires OD or OS laterality')
        if len(volume.meta['bscan_meta']) != volume.size_z:
            raise ValueError('XML B-scan metadata count disagrees with volume shape')
        own, reference = volume.meta.get('e2e_acquisition_key'), volume.meta.get('e2e_reference_key')
        external = reference is not None and reference != own
        reference_id = ctx.reference_id
        if reference is not None and reference == own and reference_id is None:
            reference_id = ctx.series_id
        if external and reference_id is None:
            raise ValueError('External E2E reference requires a receiving-database reference_id')
        if external and reference_id == ctx.series_id:
            raise ValueError('External reference cannot use the acquisition destination series_id')
        if ctx.align_followup_to_reference and reference_id != ctx.series_id:
            raise NotImplementedError('HEYEX XML follow-up alignment is not yet reproduced; use an explicit supported export setting')
        visit_date = ctx.visit_date if ctx.visit_date is not None else volume.meta.get('visit_date')
        if visit_date is None:
            raise ValueError('XML visit_date is required from source metadata or export context')
        patient = volume.meta.get('patient', {})
        if patient.get('dob') is None:
            raise ValueError('XML patient birth date is required from source metadata')
        root = ET.Element('HEDX')
        body = ET.SubElement(root, 'BODY')
        _add(body, 'ExportType', 'Data')
        software = ET.SubElement(body, 'SWVersion')
        _add(software, 'Name', ctx.software_name)
        _add(software, 'Version', ctx.software_version)
        person = ET.SubElement(body, 'Patient')
        if ctx.patient_uids:
            uids = ET.SubElement(person, 'PatientUIDList')
            _add(uids, 'NumUIDs', len(ctx.patient_uids))
            for source, uid in ctx.patient_uids:
                item = ET.SubElement(uids, 'PatientUID')
                _add(item, 'Source', source)
                _add(item, 'UID', uid)
        _add(person, 'ID', ctx.pid)
        _add(person, 'PatientID', ctx.patient_id if ctx.patient_id is not None else patient.get('patient_id', ''))
        _add(person, 'LastName', patient.get('lastname', ''))
        _add(person, 'FirstNames', patient.get('firstname', ''))
        _date(person, 'Birthdate', patient['dob'])
        _add(person, 'Sex', patient.get('sex'))
        study = ET.SubElement(person, 'Study')
        _add(study, 'StudyUID', ctx.study_uid)
        _add(study, 'ID', ctx.study_id)
        _date(study, 'StudyDate', visit_date)
        _add(study, 'Operator', ctx.operator)
        _add(study, 'ReferringPhysician', ctx.referring_physician if ctx.referring_physician is not None else '')
        series = ET.SubElement(study, 'Series')
        _add(series, 'SeriesUID', ctx.series_uid)
        _add(series, 'ID', ctx.series_id)
        _add(series, 'ExaminedStructure', ctx.examined_structure)
        _add(series, 'Modality', 'OCT')
        _add(series, 'ModalityProcedure', ctx.modality_procedure)
        _add(series, 'Type', ctx.series_type)
        _add(series, 'Laterality', 'R' if volume.laterality == 'OD' else 'L')
        _add(series, 'ProgID', volume.meta.get('prog_id'))
        if reference_id is not None:
            ref = ET.SubElement(series, 'ReferenceSeries')
            _add(ref, 'SeriesUID', ctx.reference_uid if reference_id != ctx.series_id else ctx.series_uid)
            _add(ref, 'ID', reference_id)
            reference_date = ctx.reference_exam_date
            if reference_date is None and reference_id == ctx.series_id:
                reference_date = volume.meta.get('exam_time')
            if reference_date is None:
                raise ValueError('XML reference_exam_date is required for the referenced acquisition')
            _date(ref, 'ExamDate', reference_date)
        if ctx.equipment:
            if ctx.equipment.keys() - {'Manufacturer', 'ManufacturerModelName', 'AQMName', 'AQMVersion'}:
                raise ValueError('Unsupported XML equipment context field')
            equipment = ET.SubElement(series, 'GeneralEquipment')
            for tag in ('Manufacturer', 'ManufacturerModelName'):
                _add(equipment, tag, ctx.equipment.get(tag))
            if 'AQMName' in ctx.equipment or 'AQMVersion' in ctx.equipment:
                aqm = ET.SubElement(equipment, 'AQMVersion')
                _add(aqm, 'Name', ctx.equipment.get('AQMName'))
                _add(aqm, 'Version', ctx.equipment.get('AQMVersion'))
        if ctx.oct_field_size is not None:
            field_size = ET.SubElement(series, 'OCTFieldSize')
            for tag, value in zip(('Width', 'Height'), ctx.oct_field_size, strict=True):
                _add(field_size, tag, self.number(value, 1))
        _add(series, 'NumImages', volume.size_z + 1)
        self.image(series, 0)
        for index in range(volume.size_z):
            self.image(series, index + 1)
        return root

    @staticmethod
    def number(value, precision):
        if not np.isfinite(value):
            raise ValueError('XML geometry and acquisition scalars must be finite')
        return f'{value:.{precision}f}'

    def image(self, series, number):
        volume, ctx = self.volume, self.context
        localizer = number == 0
        meta = volume.localizer.meta if localizer else volume.meta['bscan_meta'][number - 1]
        shape = volume.localizer.shape if localizer else volume.shape[1:]
        node = ET.SubElement(series, 'Image')
        _add(node, 'ID', number)
        _add(node, 'ImageNumber', number)
        _add(node, 'Laterality', 'R' if volume.laterality == 'OD' else 'L')
        clock = ctx.localizer_time if localizer else meta.get('acquisitionTime')
        if clock is None:
            clock = meta.get('heyex_xml_acquisition_time')
        _time(node, clock, ctx.utc_bias_minutes)
        kind = ET.SubElement(node, 'ImageType')
        _add(kind, 'Type', 'LOCALIZER' if localizer else 'OCT')
        if localizer:
            _add(kind, 'LightSource', 'IR' if meta.get('modality') == 'NIR' else meta.get('modality'))
        acquisition = ET.SubElement(node, 'OphthalmicAcquisitionContext')
        _add(acquisition, 'Width', shape[1])
        _add(acquisition, 'Height', shape[0])
        for axis in ('x', 'y'):
            scale = meta['scale_' + axis] if localizer else volume.meta['scale_' + axis]
            if scale <= 0:
                raise ValueError('XML pixel scales must be positive')
            encoded_scale = self.number(scale, ctx.scale_precision)
            if float(encoded_scale) <= 0:
                raise ValueError('XML scale_precision rounds a pixel scale to zero')
            _add(acquisition, 'Scale' + axis.upper(), encoded_scale)
        values = dict(meta.get('heyex_xml_acquisition', {}))
        if localizer:
            if meta.get('field_size') is not None:
                values.setdefault('Angle', meta['field_size'])
            if meta.get('scan_focus') is not None:
                values.setdefault('Focus', self.number(meta['scan_focus'], 2))
            if meta.get('num_averaged') is not None:
                values.setdefault('NumAve', meta['num_averaged'])
            values.update(ctx.localizer_acquisition)
            order = ('Angle', 'Focus', 'Resolution', 'AutoSensorGain', 'SensorGain', 'NumAve', 'FixationTarget')
        else:
            if meta.get('art_mode') is not None:
                values.setdefault('NumAve', meta['art_mode'])
            quality = meta.get('quality_ui')
            if quality is None:
                quality = meta.get('quality')
            if quality is not None and np.isfinite(quality) and quality < 3e38:
                values.setdefault('ImageQuality', self.number(quality, 0))
            values.update(ctx.oct_acquisition)
            order = ('Resolution', 'NumAve', 'ImageQuality', 'PositionWithinTolerance', 'EDI', 'EVI')
        if values.keys() - set(order):
            raise ValueError('Unsupported XML acquisition context field')
        for tag in order:
            _add(acquisition, tag, values.get(tag))
        if not localizer:
            if meta['pos_unit'] != 'mm':
                raise ValueError('XML B-scan positions must be calibrated in mm')
            for tag, key in [('Start', 'start_pos'), ('End', 'end_pos')]:
                point = ET.SubElement(ET.SubElement(acquisition, tag), 'Coord')
                for axis, value in zip(('X', 'Y'), meta[key], strict=True):
                    _add(point, axis, self.number(value, ctx.position_precision))
        data = ET.SubElement(node, 'ImageData')
        _add(data, 'Extension', 'TIF')
        _add(data, 'ExamURL', ctx.exam_url_prefix + self.names[number])
        if not localizer and self.include_segmentation and volume.layers:
            segmentations = ET.SubElement(node, 'Segmentation')
            _add(segmentations, 'NumSegmentations', len(volume.layers))
            for name, layer in volume.layers.items():
                heights = np.asarray(layer.data[number - 1], dtype=np.float32)
                if heights.shape != (volume.size_x,):
                    raise ValueError('XML layer width disagrees with the B-scan')
                line = ET.SubElement(segmentations, 'SegLine')
                _add(line, 'Name', name)
                for tag, key in [('Enabled', 'heyex_xml_enabled'), ('Manual', 'manual')]:
                    _add(line, tag, _layer_flag(layer.meta.get(key), number - 1, volume.size_z))
                array = ET.SubElement(line, 'Array', {'n': str(volume.size_x), 'type': 'real'})
                array.text = ' '.join('3e+038' if not np.isfinite(height) or height >= 3e38
                                      else format(float(height), '.9g') for height in heights)

    @property
    def bytes(self):
        return serialize_heyex_xml(self.document())

    def write(self, path):
        from eyepy.io.he._xml_tiff import encode_heyex_tiff

        path = Path(path)
        if path.exists():
            raise FileExistsError('XML export destination already exists')
        document = self.bytes
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.', dir=path.parent))
        try:
            write_file(temporary / self.filename, lambda output: output.write(document))
            for number, name in enumerate(self.names):
                pixels = self.volume.localizer.data if number == 0 else self.volume[number - 1].data
                encoded = encode_heyex_tiff(pixels)
                write_file(temporary / name, lambda output: output.write(encoded))
            if path.exists():
                raise FileExistsError('XML export destination already exists')
            temporary.rename(path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
