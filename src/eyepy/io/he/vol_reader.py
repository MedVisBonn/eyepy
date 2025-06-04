"""Inspired by:

https://github.com/ayl/heyexReader/blob/master/heyexReader/volReader.py
https://github.com/FabianRathke/octSegmentation/blob/master/collector/HDEVolImporter.m
"""
from __future__ import annotations

import datetime
import logging
from typing import Literal
import warnings

import construct as cs
import numpy as np
import skimage
from skimage import transform

from eyepy import __version__
from eyepy import EyeBscanMeta
from eyepy import EyeEnface
from eyepy import EyeEnfaceMeta
from eyepy import EyeVolume
from eyepy import EyeVolumeMeta
from eyepy.core.eyebscan import EyeBscan
from eyepy.core.utils import to_vol_intensity
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import Bscan
from eyepy.io.utils import FloatDate
from eyepy.io.utils import IntDate
from eyepy.io.utils import Localizer
from eyepy.io.utils import Segmentations

logger = logging.getLogger('eyepy.io.HeVolReader')

bscan_format = cs.Struct(
    'version' /
    cs.Default(cs.PaddedString(12, 'ascii'), f'ep-v{__version__}'.ljust(12)) *
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
    'layer_segmentations' / Segmentations *
    'Layer segmentations for num_seg Layers. If layers are not segmented the data is empty',
    '__empty' / cs.Padding(cs.this.bscan_hdr_size - 256 -
                           cs.this.num_seg * 4 * cs.this._.size_x) *
    "B-scan data starts after 'bscan_hdr_size' bytes from the start of the"
    'B-scan. This header contains 256 bytes for general information, then "num_seg" '
    'layer segmentations each composed size_x * 4 bytes. Then we have padding'
    'to fill the bscan_hdr_size.',
    'data' / Bscan * 'B-scan data',
)

vol_format = cs.Struct(
    'version' /
    cs.Default(cs.PaddedString(12, 'ascii'), f'ep-v{__version__}'.ljust(12)) *
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
    'scan_pattern' / cs.Default(cs.Int32sl, 0) * """Scan pattern type:
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
        self.path = path
        self.intensity_transform = TYPE_INTENSITY_MAPPING.get(type, 'vol')
        with open(self.path, 'rb') as vol_file:
            self.parsed_file = vol_format.parse_stream(vol_file)

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
            par_algorithm='sum',
        )


class HeVolWriter:

    def __init__(self, volume: EyeVolume):
        self.volume = volume

    def write(self, path):
        with open(path, 'wb') as mf:
            mf.write(self.bytes)

    @property
    def bytes(self):
        return vol_format.build(
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
                 exam_time=datetime.datetime.combine(
                     self.volume.meta['exam_time'], datetime.time()),
                 scan_pattern=0,
                 id='unknown',
                 reference_id='unknown',
                 pid=self.volume.meta['patient']['pid'],
                 patient_id=self.volume.meta['patient']['patient_id'],
                 dob=datetime.datetime.combine(
                     self.volume.meta['patient']['dob'], datetime.time()),
                 vid=0,
                 visit_id='unknown',
                 visit_date=datetime.datetime.combine(
                     self.volume.meta['visit_date'], datetime.time()),
                 grid_type=0,
                 grid_offset=0,
                 grid_type1=0,
                 grid_offset1=0,
                 prog_id='unknown',
                 localizer=skimage.util.img_as_ubyte(
                     self.volume.localizer.data),
                 bscans=self._bscan_dicts))

    @property
    def _bscan_dicts(self):
        return [
            dict(bscan_hdr_size=256 + 17 * 4 * bscan.volume.size_x,
                 start_x=bscan.meta['start_pos'][0],
                 start_y=bscan.meta['start_pos'][1],
                 end_x=bscan.meta['end_pos'][0],
                 end_y=bscan.meta['end_pos'][1],
                 num_seg=len(bscan.layers),
                 quality=bscan.meta['quality'],
                 shift=0,
                 iv_transformation=[0, 0, 0, 0, 0, 0],
                 layer_segmentations=self._segmentations_from_bscan(bscan),
                 data=to_vol_intensity(bscan.data.astype(np.float32)))
            for bscan in self.volume
        ]

    def _segmentations_from_bscan(self, bscan: EyeBscan) -> np.ndarray:
        """"""
        segs = np.zeros((17, bscan.volume.size_x), dtype=np.float32)
        for layer in bscan.layers:
            if layer.name in SEG_MAPPING:
                segs[SEG_MAPPING[layer.name]] = layer.data
            else:
                warnings.warn(
                    f'Unknown layer name: {layer.name}. Skipping. \n VOL format only supports layers with the following names: {list(SEG_MAPPING.keys())}'
                )
        return segs
