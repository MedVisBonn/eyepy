"""Inspired by:

https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""

from __future__ import annotations

import datetime
import io
import logging
import struct
from typing import Literal
import warnings

import construct as cs
import numpy as np
import skimage

from eyepy import EyeBscanMeta
from eyepy import EyeEnface
from eyepy import EyeEnfaceMeta
from eyepy import EyeVolume
from eyepy import EyeVolumeMeta
from eyepy.core.eyebscan import EyeBscan
from eyepy.core.utils import display_to_vol_linear_approx
from eyepy.io.he._vol_export import HeVolExportContext
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import Bscan
from eyepy.io.utils import FloatDate
from eyepy.io.utils import IntDate
from eyepy.io.utils import Localizer
from eyepy.io.utils import Segmentations

logger = logging.getLogger('eyepy.io.HeVolReader')

bscan_format = cs.Struct(
    'version' /
    cs.Default(cs.PaddedString(12, 'ascii'), 'HSF-BS-103') *
    'Version identifier',
    'bscan_hdr_size' / cs.Int32ul *
    'Size of the B-Scan header in bytes. It is identical to the same value of the file header.',
    'start_x' / cs.Float64l *
    "X-Coordinate of the B-Scan's start point in mm.",
    'start_y' / cs.Float64l *
    "Y-Coordinate of the B-Scan's start point in mm.",
    'end_x' / cs.Float64l *
    "X-Coordinate of the B-Scan's end point in mm. For circle scans, this is the X-Coordinate of the circle's center point.",
    'end_y' / cs.Float64l *
    "Y-Coordinate of the B-Scan's end point in mm. For circle scans, this is the Y-Coordinate of the circle's center point.",
    'num_seg' / cs.Int32ul * 'Number of segmentation vectors',
    'seg_offset' / cs.Default(cs.Int32ul, 256) *
    'Offset of the array of segmentation vectors relative to the beginning of this B-Scan header.',
    'quality' / cs.Float32l *
    'Image quality measure. If this value does not exist, its value is set to INVALID.',
    'shift' / cs.Int32ul *
    'Horizontal shift (in # of A-Scans) of the classification band against the segmentation lines (for circular scan only).',
    'iv_transformation' / cs.Float32l[6] *
    'Intra volume transformation matrix. The values are only available for volume and radial scans and if alignment is turned off, otherwise the values are initialized to 0.',
    '__empty' / cs.Padding(168) * 'Spare bytes for future use.',
    # cs.Probe(cs.this.bscan_hdr.sizeof(cs.this)),
    '_seg_padding' / cs.Padding(cs.this.seg_offset - 256),
    'layer_segmentations' / Segmentations *
    'Layer segmentations for num_seg Layers. If layers are not segmented the data is empty',
    '__empty' / cs.Padding(cs.this.bscan_hdr_size - cs.this.seg_offset -
                           cs.this.num_seg * 4 * cs.this._.size_x) *
    "B-scan data starts after 'bscan_hdr_size' bytes from the start of the"
    'B-scan. This header contains 256 bytes for general information, then "num_seg" '
    'layer segmentations each composed size_x * 4 bytes. Then we have padding'
    'to fill the bscan_hdr_size.',
    'data' / Bscan * 'B-scan data',
)

vol_format = cs.Struct(
    'version' /
    cs.Default(cs.PaddedString(12, 'ascii'), 'HSF-OCT-103') *
    'Version identifier: HSF-OCT-xxx',
    'size_x' / cs.Int32ul *
    'Number of A-Scans in each B-Scan, i.e. B-Scan width in pixel',
    'n_bscans' / cs.Int32ul * 'Number of B-Scans in OCT scan',
    'size_y' / cs.Int32ul *
    'Number of samples in an A-Scan, i.e. B-Scan height in pixel',
    'scale_x' / cs.Float64l * 'Width of a B-Scan pixel in mm',
    'distance' / cs.Float64l * 'Distance between two adjacent B-Scans in mm',
    'scale_y' / cs.Float64l * 'Height of a B-Scan pixel in mm',
    'size_x_slo' / cs.Int32ul * 'Width of the SLO image in pixel',
    'size_y_slo' / cs.Int32ul * 'Height of the SLO image in pixel',
    'scale_x_slo' / cs.Float64l * 'Width of a pixel in the SLO image in mm',
    'scale_y_slo' / cs.Float64l * 'Height of a pixel in the SLO image in mm',
    'field_size_slo' / cs.Default(cs.Int32ul, 0) *
    'Horizontal field size of the SLO image in dgr',
    'scan_focus' / cs.Default(cs.Float64l, 0) * 'Scan focus in dpt',
    'scan_position' / cs.Default(cs.PaddedString(4, 'ascii'), ''.ljust(4)) *
    "Examined eye (zero terminated string). 'OS': Left eye; 'OD': Right eye",
    'exam_time' /
    cs.Default(IntDate, datetime.datetime(year=2000, month=1, day=1)) *
    'Examination time. The structure holds an unsigned 64-bit date and time value. '
    'It is encoded as 100ns units since beginning of January 1, 1601',
    'scan_pattern' / cs.Default(cs.Int32sl, 0) *\
                                                 """Scan pattern type:
    0 = Unknown pattern,
    1 = Single line scan (one B-Scan only),
    2 = Circular scan (one B-Scan only),
    3 = Volume scan in ART mode,
    4 = Fast volume scan,
    5 = Radial scan (aka. star pattern)""",
    'bscan_hdr_size' / cs.Default(cs.Int32ul, 61440) *
    'Size of the Header preceding each B-Scan in bytes',
    'id' / cs.PaddedString(16, 'ascii') *
    'Unique identifier of this OCT-scan (zero terminated string). This is'
    'identical to the number <SerID> that is part of the file name.'
    'Format: n[.m] n and m are numbers. The extension .m exists only for '
    'ScanPattern 1 and 2. Examples: 2390, 3433.2',
    'reference_id' / cs.PaddedString(16, 'ascii') *
    'Unique identifier of the reference OCT-scan (zero terminated string).'
    'Format: see ID, This ID is only present if the OCT-scan is part of a '
    'progression otherwise this string is empty. For the reference scan of '
    'a progression ID and ReferenceID are identical.',
    'pid' / cs.Default(cs.Int32ul, 0) * 'Internal patient ID used by HEYEX.',
    'patient_id' /
    cs.Default(cs.PaddedString(24, 'ascii'), 'unknown'.ljust(24)) *
    'User-defined patient ID (zero terminated string).',
    'dob' / FloatDate * "Patient's date of birth",
    'vid' / cs.Default(cs.Int32ul, 0) * 'Internal visit ID used by HEYEX.',
    'visit_id' /
    cs.Default(cs.PaddedString(24, 'ascii'), 'unknown'.ljust(24)) *
    'User-defined visit ID (zero terminated string). This ID can be defined '
    'in the Comment-field of the Diagnosis-tab of the Examination Data '
    'dialog box. The VisitID must be defined in the first row of the '
    "comment field. It has to begin with an '#' and ends with any "
    'white-space character. It can contain up to 23 alphanumeric '
    "characters (excluding the '#')",
    'visit_date' / FloatDate *
    'Date the visit took place. Identical to the date of an examination tab in HEYEX.',
    'grid_type' / cs.Int32ul *
    'Type of grid used to derive thickness data. 0 No thickness data available, >0 Type'
    ' of grid used to derive thickness values. Thickness data is only available for '
    'ScanPattern 3 and 4.',
    'grid_offset' / cs.Int32ul *
    'File offset of the thickness data in the file. If GridType is 0, GridOffset is 0.',
    'grid_type1' / cs.Int32ul *
    'Type of 2nd grid used to derive a 2nd set of thickness data.',
    'grid_offset1' / cs.Int32ul *
    'File offset of the 2nd thickness data set in the file.',
    'prog_id' / cs.PaddedString(34, 'ascii') *
    'Internal progression ID (zero terminated string). All scans of the '
    'same progression share this ID.',
    '__empty' / cs.Padding(1790) *
    'Spare bytes for future use. Initialized to 0.',
    'localizer' / Localizer * 'NIR localizer image',
    'bscans' / bscan_format[cs.this.n_bscans],
)

# Building a large volume must not retain every generated B-scan array in
# Construct's return value. Parsing still uses vol_format's ordinary Array.
vol_write_format = cs.Struct(
    *vol_format.subcons[:-1],
    'bscans' / cs.Array(cs.this.n_bscans, bscan_format, discard=True))

# PR1 and EZ map to 14 and PR2 and IZ map to 15. Hence both names can be used
# to access the same data
SEG_MAPPING = {
    'ILM': 0,
    'BM': 1,
    'RNFL': 2,
    'GCL': 3,
    'IPL': 4,
    'INL': 5,
    'OPL': 6,
    'ONL': 7,
    'ELM': 8,
    'IOS': 9,
    'OPT': 10,
    'CHO': 11,
    'VIT': 12,
    'ANT': 13,
    'PR1': 14,
    'PR2': 15,
    'RPE': 16,
    'IPL+': 17,
    'IPL-': 18,
}

SLAB_MAPPING = {
    'NFLVP': ('ILM', 'RNFL'),
    'SVP': ('RNFL', 'IPL-'),
    'ICP': ('IPL-', 'IPL+'),
    'DCP': ('IPL+', 'OPL'),
    'SVC': ('ILM', 'IPL-'),
    'DVC': ('IPL-', 'OPL'),
    'AVAC': ('OPL', 'BM'), # Avascular Complex
    'RET': ('ILM', 'BM'), # Retina
}

TYPE_INTENSITY_MAPPING = {
    'oct': 'vol',
    'octa': 'angio',
}


class HeVolReader:

    def __init__(self, path, type: Literal['oct', 'octa'] = 'oct'):
        from eyepy.io.he._native import SourceFile
        from eyepy.io.he._vol_native import capture_layout

        self.path = path
        self._source = SourceFile.capture(path)
        self.intensity_transform = TYPE_INTENSITY_MAPPING.get(type, 'vol')
        with open(self.path, 'rb') as vol_file:
            self.parsed_file = vol_format.parse_stream(vol_file)
        self._source.verify()
        self._native_layout = capture_layout(self)

    @property
    def volume(self) -> EyeVolume:
        ## Check if scan pattern is supported by EyeVolume
        if self.parsed_file.scan_pattern == 2:
            msg = f'The EyeVolume object does not support scan pattern 2 (one Circular B-scan).'
            raise ValueError(msg)
        elif self.parsed_file.scan_pattern == 5:
            msg = f'The EyeVolume object does not support scan pattern 5 (Radial scan - star pattern).'
            raise ValueError(msg)

        data = np.stack([bscan.data for bscan in self.parsed_file.bscans],
                        axis=0)
        volume_meta = self.meta
        localizer = self.localizer
        volume = EyeVolume(
            data=data,
            meta=volume_meta,
            localizer=localizer,
            transformation=_compute_localizer_oct_transform(
                volume_meta, localizer.meta, data.shape),
        )

        layer_height_maps = self.layers
        for name, i in SEG_MAPPING.items():
            if i >= layer_height_maps.shape[0]:
                logger.warning(
                    'The volume contains less layers than expected. The naming might not be correct.'
                )
                break
            volume.add_layer_annotation(layer_height_maps[i], name=name)

        for name, (top, bottom) in SLAB_MAPPING.items():
            slab_meta = {
                'name': name,
                'top_layer': top,
                'bottom_layer': bottom
            }
            volume.add_slab_annotation(meta=slab_meta)

        return volume

    @property
    def layers(self):
        la = [b.layer_segmentations for b in self.parsed_file.bscans]
        n_layers = np.unique([len(l) for l in la])
        if len(n_layers) > 1:
            max_layers = np.max(n_layers)
            la = [
                np.pad(l, ((0, max_layers - len(l)), (0, 0)),
                       'constant',
                       constant_values=np.nan) for l in la
            ]
        layers = np.stack(la, axis=0)
        layers[layers >= 3.0e+38] = np.nan
        # Currently the shape is (n_bscans, n_layers, width). Swap the first two axes
        # to get (n_layers, n_bscans, width)
        return np.swapaxes(layers, 0, 1)

    @property
    def localizer_meta(self) -> EyeEnfaceMeta:
        return EyeEnfaceMeta(
            scale_x=self.parsed_file.scale_x_slo,
            scale_y=self.parsed_file.scale_y_slo,
            scale_unit='mm',
            modality='NIR',
            laterality=self.parsed_file.scan_position,
            field_size=self.parsed_file.field_size_slo,
            scan_focus=self.parsed_file.scan_focus,
            visit_date=self.parsed_file.visit_date,
            exam_time=self.parsed_file.exam_time,
        )

    @property
    def localizer(self) -> EyeEnface:
        localizer = self.parsed_file.localizer
        return EyeEnface(localizer, self.localizer_meta)

    @property
    def bscan_meta(self) -> list[EyeBscanMeta]:
        return [
            EyeBscanMeta(
                quality=b.quality,
                num_seg=b.num_seg,
                seg_offset=b.seg_offset,
                shift=b.shift,
                iv_transformation=list(b.iv_transformation),
                start_pos=(b.start_x, b.start_y),
                end_pos=(b.end_x, b.end_y),
                pos_unit='mm',
            ) for b in self.parsed_file.bscans
        ]

    @property
    def meta(self) -> EyeVolumeMeta:
        bscan_meta = self.bscan_meta
        return EyeVolumeMeta(
            scale_x=self.parsed_file.scale_x,
            scale_y=self.parsed_file.scale_y,
            scale_z=self.parsed_file.distance,
            scale_unit='mm',
            laterality=self.parsed_file.scan_position,
            visit_date=self.parsed_file.visit_date,
            exam_time=self.parsed_file.exam_time,
            bscan_meta=bscan_meta,
            intensity_transform=self.intensity_transform,
            intensity_encoding='vol_linear',
            par_algorithm='sum',
            patient=dict(pid=self.parsed_file.pid,
                         patient_id=self.parsed_file.patient_id,
                         dob=self.parsed_file.dob),
            id=self.parsed_file.id,
            reference_id=self.parsed_file.reference_id,
            vid=self.parsed_file.vid,
            visit_id=self.parsed_file.visit_id,
            prog_id=self.parsed_file.prog_id,
            scan_pattern=self.parsed_file.scan_pattern,
            bscan_hdr_size=self.parsed_file.bscan_hdr_size,
            heyex_native_dates=dict(
                exam_time=self.parsed_file.exam_time,
                exam_time_ticks=struct.unpack('<Q', self._source.read(88, 8))[0],
                dob=self.parsed_file.dob,
                dob_days=struct.unpack('<d', self._source.read(164, 8))[0],
                visit_date=self.parsed_file.visit_date,
                visit_date_days=struct.unpack('<d', self._source.read(200, 8))[0]),
        )


class _BscanRecords:

    def __init__(self, writer):
        self.writer = writer

    def __len__(self):
        return self.writer.volume.size_z

    def __iter__(self):
        return self.writer._iter_bscan_dicts()


class HeVolWriter:
    """Export supported linear OCT values without a display round trip.

    Display-only inputs require allow_approximate=True. The approximate
    inverse uses the stored display greys, independently of
    visualization settings. Raw E2E inputs without a supported transform
    and unknown encodings cannot be exported.
    """

    def __init__(self, volume: EyeVolume | HeVolReader, *, allow_approximate: bool = False,
                 export_context: HeVolExportContext | None = None):
        self.reader = volume if isinstance(volume, HeVolReader) else None
        self.volume = None if self.reader is not None else volume
        self.allow_approximate = allow_approximate
        if export_context is not None and not isinstance(export_context, HeVolExportContext):
            raise TypeError('export_context must be a HeVolExportContext')
        if self.reader is not None and export_context is not None:
            raise ValueError('Use export_context with an EyeVolume; native reader edits use parsed_file')
        self.export_context = export_context

    def _validate_encoding(self):
        if self.volume.linear_data_available:
            return
        if self.volume.intensity_encoding == 'display':
            if self.allow_approximate:
                return
            raise ValueError(
                'Display-only input cannot recover original linear intensities. '
                'Use allow_approximate=True to export representative VOL values.')
        self.volume._require_linear_data()

    def write(self, path, *, overwrite=False):
        from eyepy.io.he._native import write_file
        from eyepy.io.he._vol_native import patches

        if self.reader is not None:
            source = self.reader._source
            write_file(path, lambda output: source.rewrite(output, patches(self.reader)),
                       sources=[source], overwrite=overwrite)
        else:
            self._validate_encoding()
            metadata = self._export_metadata()
            write_file(path, lambda output: self._write_generated(output, metadata),
                       overwrite=overwrite)

    @staticmethod
    def _date(value):
        """Normalize metadata dates to the ISO strings used by VOL adapters."""
        if isinstance(value, datetime.date):
            value = value.isoformat()
        value = datetime.datetime.fromisoformat(value)
        if value.tzinfo is not None:
            value = value.astimezone(datetime.UTC).replace(tzinfo=None)
        return value.isoformat()

    @property
    def bytes(self):
        if self.reader is not None:
            from eyepy.io.he._vol_native import patches
            return self.reader._source.build(patches(self.reader))
        self._validate_encoding()
        metadata = self._export_metadata()
        output = io.BytesIO()
        self._write_generated(output, metadata)
        return output.getvalue()

    def _write_generated(self, output, metadata):
        vol_write_format.build_stream(
            dict(size_x=self.volume.size_x,
                 size_y=self.volume.size_y,
                 n_bscans=self.volume.size_z,
                 scale_x=self.volume.scale_x,
                 scale_y=self.volume.scale_y,
                 distance=self.volume.scale_z,
                 size_x_slo=self.volume.localizer.size_x,
                 size_y_slo=self.volume.localizer.size_y,
                 scale_x_slo=self.volume.localizer.scale_x,
                 scale_y_slo=self.volume.localizer.scale_y,
                 field_size_slo=self.volume.localizer.meta['field_size'],
                 scan_focus=self.volume.localizer.meta['scan_focus'],
                 scan_position=self.volume.localizer.meta['laterality'],
                 exam_time=metadata['exam_time'],
                 scan_pattern=metadata['scan_pattern'],
                 bscan_hdr_size=self._bscan_header_size,
                 id=metadata['id'],
                 reference_id=metadata['reference_id'],
                 pid=metadata['pid'],
                 patient_id=metadata['patient_id'],
                 dob=metadata['dob'],
                 vid=metadata['vid'],
                 visit_id=metadata['visit_id'],
                 visit_date=metadata['visit_date'],
                 grid_type=0,
                 grid_offset=0,
                 grid_type1=0,
                 grid_offset1=0,
                 prog_id=metadata['prog_id'],
                 localizer=skimage.util.img_as_ubyte(
                     self.volume.localizer.data),
                 bscans=_BscanRecords(self)), output)
        # Datetimes cannot retain native 100 ns ticks or every float64 day
        # value. Reuse those scalar encodings only while their meaning is
        # unchanged; an explicit metadata/context edit takes precedence.
        dates = self.volume.meta.get('heyex_native_dates', {})
        context = self.export_context.as_dict() if self.export_context is not None else {}
        end = output.tell()
        for name, raw_name, offset, code in (
                ('exam_time', 'exam_time_ticks', 88, '<Q'),
                ('dob', 'dob_days', 164, '<d'),
                ('visit_date', 'visit_date_days', 200, '<d')):
            if raw_name in context:
                output.seek(offset)
                output.write(struct.pack(code, context[raw_name]))
            elif name not in context and dates.get(name) is not None and raw_name in dates:
                if metadata[name] == self._date(dates[name]):
                    output.seek(offset)
                    output.write(struct.pack(code, dates[raw_name]))
        output.seek(end)

    def _export_metadata(self):
        meta = self.volume.meta
        patient = meta.get('patient', {})
        values = dict(id=meta.get('id', ''), reference_id=meta.get('reference_id', ''),
                      pid=patient.get('pid'), patient_id=patient.get('patient_id', ''),
                      vid=meta.get('vid', 0), visit_id=meta.get('visit_id', ''),
                      prog_id=meta.get('prog_id', ''), scan_pattern=meta.get('scan_pattern', 0),
                      exam_time=meta.get('exam_time'), visit_date=meta.get('visit_date'),
                      dob=patient.get('dob'))
        if self.export_context is not None:
            values.update(self.export_context.as_dict())
            for name, raw_name, code, adapter in (
                    ('exam_time', 'exam_time_ticks', '<Q', IntDate),
                    ('visit_date', 'visit_date_days', '<d', FloatDate)):
                if values.get(raw_name) is not None:
                    if getattr(self.export_context, name) is not None:
                        raise ValueError(f'Supply either {name} or {raw_name}, not both')
                    try:
                        values[name] = adapter.parse(struct.pack(code, values[raw_name]))
                    except (ValueError, OverflowError, struct.error) as error:
                        raise ValueError(f'Invalid receiving-database {raw_name}') from error
        if self.volume.intensity_encoding == 'e2e_raw':
            if not values['id']:
                raise ValueError('E2E conversion requires export_context with receiving-database identifiers')
            if 'scan_pattern' not in meta and (self.export_context is None or self.export_context.scan_pattern is None):
                raise ValueError('scan_pattern is required in the E2E receiving-database export_context')
            reference = meta.get('e2e_reference_key')
            if reference is not None and (self.export_context is None or self.export_context.reference_id is None):
                if reference == meta.get('e2e_acquisition_key'):
                    values['reference_id'] = values['id']
                else:
                    raise ValueError('reference_id is required to map the external E2E reference into the receiving database')
        for name in ('id', 'reference_id', 'patient_id', 'visit_id', 'prog_id'):
            if values[name] is None:
                values[name] = ''
        for name in ('pid', 'scan_pattern', 'exam_time', 'visit_date', 'dob'):
            if values[name] is None:
                raise ValueError(f'{name} is required for VOL export; supply source metadata or export_context')
        for name in ('exam_time', 'visit_date', 'dob'):
            values[name] = self._date(values[name])
        return values

    @property
    def _num_segments(self):
        source_counts = [meta.get('num_seg', 0) for meta in self.volume.meta['bscan_meta']]
        layer_counts = [SEG_MAPPING[name] + 1 for name in self.volume.layers if name in SEG_MAPPING]
        return max([0, *source_counts, *layer_counts])

    @property
    def _bscan_header_size(self):
        required = max(meta.get('seg_offset', 256) for meta in self.volume.meta['bscan_meta'])
        required += self._num_segments * 4 * self.volume.size_x
        return max(self.volume.meta.get('bscan_hdr_size', 80 * self.volume.size_x), required)

    def _iter_bscan_dicts(self):
        self._validate_encoding()
        header_size = self._bscan_header_size
        count = self._num_segments
        matrices = getattr(self.volume.bscan_transform, 'matrices', None)
        return (
            dict(bscan_hdr_size=header_size,
                 start_x=bscan.meta['start_pos'][0],
                 start_y=bscan.meta['start_pos'][1],
                 end_x=bscan.meta['end_pos'][0],
                 end_y=bscan.meta['end_pos'][1],
                 num_seg=count,
                 seg_offset=bscan.meta.get('seg_offset', 256),
                 quality=bscan.meta['quality_ui'] if 'quality_ui' in bscan.meta else
                 bscan.meta.get('quality', np.finfo(np.float32).max),
                 shift=bscan.meta.get('shift', 0),
                 iv_transformation=(matrices[index].ravel().tolist() if matrices is not None else
                                    bscan.meta.get('iv_transformation', [0] * 6)),
                 layer_segmentations=self._segmentations_from_bscan(bscan),
                 data=(bscan.linear_data.astype('<f4', copy=False)
                       if self.volume.linear_data_available else
                       display_to_vol_linear_approx(bscan.raw_data)))
            for index, bscan in enumerate(self.volume)
        )

    def _segmentations_from_bscan(self, bscan: EyeBscan) -> np.ndarray:
        """"""
        segs = np.full((self._num_segments, bscan.volume.size_x),
                       np.finfo(np.float32).max, dtype='<f4')
        # B-scan layer views are lazy; iterating their cache can omit every
        # layer unless a caller happened to access them before exporting.
        for layer in self.volume.layers.values():
            if layer.name in SEG_MAPPING:
                heights = np.asarray(layer.data[bscan.index], dtype='<f4')
                segs[SEG_MAPPING[layer.name]] = np.where(np.isnan(heights), np.finfo(np.float32).max, heights)
            else:
                warnings.warn(
                    f'Unknown layer name: {layer.name}. Skipping. \n VOL format only supports layers with the following names: {list(SEG_MAPPING.keys())}'
                )
        return segs
