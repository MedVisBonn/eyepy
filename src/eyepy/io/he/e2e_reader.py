from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from copy import deepcopy
import dataclasses
import datetime
from io import BufferedReader
import logging
import math
from pathlib import Path
import re
import struct
import sys
from textwrap import indent
import traceback
from typing import Any, Optional, Union

import construct as cs
import numpy as np
from skimage.transform import AffineTransform
from skimage.transform import warp

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.eyevolume import EyeVolume
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import find_float
from eyepy.io.utils import find_int
from eyepy.io.utils import get_bscan_spacing

from .e2e_calibration import E2EOpticalModel
from .e2e_calibration import inverse_localizer_registration
from .e2e_format import ContainerHeader
from .e2e_format import containerheader_format
from .e2e_format import DataContainer
from .e2e_format import datacontainer_format
from .e2e_format import e2e_format
from .e2e_format import Type10004
from .e2e_format import type10004_format
from .e2e_format import Type10025
from .e2e_format import TypesEnum
from .vol_reader import SEG_MAPPING

logger = logging.getLogger(__name__)

# Occurence of type ids. This is used by the inspect function.
type_occurence = {
    'E2EFileStructure': [0, 9011],
    'E2EPatientStructure': [6, 9, 17, 29, 31, 52, 9010],
    'E2EStudyStructure': [7, 10, 13, 30, 53, 58, 1000, 9000, 9001],
    'E2ESeriesStructure': [
        2, 3, 11, 54, 59, 61, 62, 1000, 1001, 1003, 1008, 9005, 9006, 9007,
        9008, 10005, 10009, 10010, 10011, 10013, 10020, 10025, 10090, 1073741824, 1073751824,
        1073751825, 1073751826
    ],
    'E2ESliceStructure': [
        2, 3, 5, 39, 40, 10004, 10012, 10013, 10019, 10020, 10032, 1073741824,
        1073751824, 1073751825, 1073751826
    ]
}

HEYEX_TIMEZONE_SUFFIX = 'UTC+1'


def _format_version(raw: bytes) -> str:
    return '.'.join(str(part) for part in raw)


def _format_px_mm(size_px: int, scale_um_per_px: float | None) -> str:
    if scale_um_per_px is None:
        return f'{size_px} pixels'
    return f'{size_px} pixels ({size_px * scale_um_per_px / 1000:.1f} mm)'


def _format_scan_focus(value_diopters: float | None) -> str | None:
    if value_diopters is None:
        return None
    return f'{value_diopters:.2f} D'


def _format_time(iso_value: str | None,
                 timezone_suffix: str = HEYEX_TIMEZONE_SUFFIX) -> str | None:
    if iso_value is None:
        return None
    dt = datetime.datetime.fromisoformat(iso_value) + datetime.timedelta(hours=1)
    return f'{dt:%H:%M:%S} ({timezone_suffix})'


def _extract_ascii_codes(raw: bytes) -> list[str]:
    seen = set()
    values = []
    for match in re.findall(rb'S[0-9]{4}[A-Z0-9-]*', raw):
        text = match.decode('ascii', errors='ignore')
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return values


class E2EStructureMixin:
    """A Mixin for shared functionality between structures in the E2E
    hierarchy."""

    def inspect(self,
                recursive: bool = False,
                ind_prefix: str = '',
                tables: bool = False) -> str:
        """Inspect the E2E structure.

        Args:
            recursive: If True inspect lower level structures recursively.
            ind_prefix: Indentation for showing information from lower level structures.
            tables: If True add markdown table overview of the contained folder types.

        Returns:
            Information about the E2E structure.
        """
        text = self._get_section_title() + '\n'
        text += self._get_section_description() + '\n'
        if tables:
            text += self._get_folder_summary() + '\n'

        if not recursive:
            return text

        for s in self.substructure.values():
            text += '\n'
            text += indent(s.inspect(recursive, ind_prefix, tables),
                           ind_prefix)
        return text

    def get_folder_data(
        self,
        folder_type: TypesEnum | int,
        offset: int = 0,
        data_construct: cs.Construct | str | None = None,
    ) -> Any:
        """Return the data of a folder type.

        Args:
            folder_type: Either one of [TypesEnum][eyepy.io.he.e2e_format.TypesEnum] or the type id (int).
            offset: Offset to the data in bytes.
            data_construct: A construct to parse the data with (Python construct package) or a string describing one of the basic constructs from the construct package like "Int8ul", or "Float32l"

        Returns:
            Parsed data or None if no folder of the given type was found.
        """
        folders: list[E2EFolder] = self.folders[folder_type]

        if len(folders) == 0:
            return None

        if data_construct is None:
            return [f.data for f in folders]
        elif type(data_construct) == str:
            data_construct = getattr(cs, data_construct)

        return [f.parse_spec(data_construct, offset) for f in folders]

    def __str__(self):
        return self.inspect()

    def _get_section_title(self) -> str:
        """Make a title for describing the structure.

        Used by the inspect function.
        """
        if not self._section_title:
            try:
                self._section_title = f'{self.__class__.__name__}({self.id})'
            except:
                self._section_title = f'{self.__class__.__name__}'

        return self._section_title

    def _get_table(self, data, structure=None) -> str:
        """Make a markdown table.

        Used by the inspect function.
        """
        if structure is None:
            structure = self.__class__.__name__
        data = [[
            f'{TypesEnum(k).name} ({k})' if k in TypesEnum else f'{k}',
            len(v),
            np.mean(v),
            np.min(v),
            np.max(v), False if k not in type_occurence[structure] else True
        ] for k, v in data.items()]
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                'pandas is required for table output. Install it with: '
                'pip install eyepy[pandas]. Alternatively, call inspect with '
                'tables=False.'
            ) from None
        text = pd.DataFrame.from_records(data,
                                         columns=[
                                             'Type', 'Count', 'Mean Size',
                                             'Min Size', 'Max Size',
                                             'described'
                                         ]).to_markdown(index=False)
        return text

    def _get_folder_summary(self) -> str:
        """Make a markdown table with folder type summary for the structure.

        Used by the inspect function.
        """
        data = defaultdict(list)
        for f_list in self.folders.values():
            for f in f_list:
                data[f.type].append(f.size)

        text = self._get_table(data)
        return text

    def _get_section_description(self) -> str:
        """Make a description for describing the structure.

        This uses the _section_description_parts attribute that can be
        defined in a structure to make a description.

        Used by the inspect function.
        """
        if not self._section_description:
            self._section_description = ''
            for part in self._section_description_parts:
                try:
                    self._section_description += f'{part[0]} {self.folders[part[1]][0].data.text[part[2]]} - '
                except:
                    pass
            self._section_description = self._section_description.rstrip(' - ')
        return self._section_description


@dataclasses.dataclass
class E2EFolder():
    """Folder data class.

    !!! Note

        Folders are created during initialization of the HeE2eReader. For accessing the data the
        respective HeE2eReader has to be used as a Context Manager. This opens the E2E file and
        allows the E2EFolder to access the data.


        ```python
        with HeE2eReader("path/to/e2e") as reader:
            folder_dict = reader.file_hierarchy.folders
            folder = folder_dict[TypesEnum.YOURTYPE][0]
            data = folder.data
        ```
    """
    patient_id: int
    study_id: int
    series_id: int
    slice_id: int
    pos: int
    start: int
    type: int
    size: int
    ind: int
    reader: HeE2eReader

    _data = None
    _header = None
    _bytes = None

    @property
    def file_object(self) -> BufferedReader:
        """Return the file object.

        This refers to the the HeE2eReader file object.
        """
        return self.reader.file_object

    @property
    def data(self) -> Any:
        """Return the data."""
        if self._data is None:
            parsed = self._parse_data()
            self._data = parsed.item
            self._header = parsed.header
        return self._data

    @property
    def header(self) -> ContainerHeader:
        """Return the data header."""
        if self._header is None:
            parsed = self._parse_data()
            self._data = parsed.item
            self._header = parsed.header
        return self._header

    def _parse_data(self) -> DataContainer:
        """Parse the data.

        This only works if the HeE2eReader is used as a Context Manager
        or during initialization of the HeE2eReader. Otherwise the E2E
        file is not open.
        """
        if self.is_inline:
            raise ValueError('Inline E2E index values have no container payload')
        self.file_object.seek(self.start)
        header = self.file_object.read(containerheader_format.sizeof())
        if len(header) != containerheader_format.sizeof():
            raise EOFError('Truncated E2E container header')
        if containerheader_format.parse(header).size != self.size:
            raise ValueError('E2E directory/container payload sizes disagree')
        # Bound GreedyBytes to this container rather than the rest of the file.
        return datacontainer_format.parse(header + self.get_bytes())

    @property
    def is_inline(self) -> bool:
        """Small directory start values encode inline data, not file offsets."""
        return self.start < 88

    def parse_spec(self, data_construct: cs.Construct, offset: int = 0) -> Any:
        """Parse a data specification.

        This only works if the HeE2eReader is used as a Context Manager or during initialization of the HeE2eReader.
        Otherwise the E2E file is not open.



        Args:
            data_construct: The construct to parse the data with. You can Constructs defined in the construct library or those defined in the [e2e_format][eyepy.io.he.e2e_format] module.
            offset: The offset in bytes, 0 by default.
        """
        b = self.get_bytes()
        return data_construct.parse(b[offset:])

    def get_bytes(self) -> bytes:
        """Return the bytes of the data.

        This only works if the HeE2eReader is used as a Context Manager
        or during initialization of the HeE2eReader. Otherwise the E2E
        file is not open.
        """
        if self._bytes is None:
            if self.is_inline:
                raise ValueError('Inline E2E index values have no container payload')
            self.file_object.seek(self.start + containerheader_format.sizeof())
            self._bytes = self.file_object.read(self.size)
            if len(self._bytes) != self.size:
                self._bytes = None
                raise EOFError('Truncated E2E container payload')
        return self._bytes


class E2ESliceStructure(E2EStructureMixin):
    """E2E Slice Structure.

    This structure contains folders with data for a single Slice/B-csan
    and provide convenience functions for accessing the data.
    """

    def __init__(self, id: int) -> None:
        self.id = id
        self.folders: dict[int | str, list[E2EFolder]] = {}
        self._meta = None
        self._bscanmeta_data = None

        # Empty so inspect() does not fail
        self.substructure = {}

    def add_folder(self, folder: E2EFolder) -> None:
        """Add a folder to the slice.

        Args:
            folder: The folder to add.
        """
        try:
            self.folders[folder.type].append(folder)
        except KeyError:
            self.folders[folder.type] = [folder]

    def get_layers(self) -> dict[int, np.ndarray]:
        """Return the layers as a dictionary of layer id and layer data."""
        layers = {}
        for layer_folder in self.folders.get(TypesEnum.layer_annotation, []):
            layers[layer_folder.data.id] = layer_folder.data.data
        return layers

    def get_layer_manual_flags(self) -> dict[int, bool]:
        """Return manual edit flags by layer id.

        For Type10019 layer annotations, the ``manual_edit_raw`` field
        indicates whether the B-scan layer was manually changed.
        """
        manual_flags = {}
        for layer_folder in self.folders.get(TypesEnum.layer_annotation, []):
            layer_data = layer_folder.data
            manual_flags[layer_data.id] = bool(layer_data.manual_edit_raw)
        return manual_flags

    def _get_bscanmeta_data(self) -> Type10004:
        if self._bscanmeta_data is None:
            folder = self.folders[TypesEnum.bscanmeta][0]
            self._bscanmeta_data = type10004_format.parse(folder.get_bytes())
        return self._bscanmeta_data

    def get_meta(self) -> EyeBscanMeta:
        """Return the slice meta data."""
        if self._meta is not None:
            return self._meta

        if len(self.folders[TypesEnum.bscanmeta]) > 1:
            logger.warning(
                'There is more than one bscanmeta object. This is not expected.'
            )

        meta = self._get_bscanmeta_data()
        meta_dict = {
            'unknown0': meta.unknown0,
            'size_y': meta.size_y,
            'size_x': meta.size_x,
            'start_x': meta.start_x,
            'start_y': meta.start_y,
            'end_x': meta.end_x,
            'end_y': meta.end_y,
            'zero1': meta.zero1,
            'unknown1': meta.unknown1,
            'scale_y': meta.scale_y,
            'unknown2': meta.unknown2,
            'zero2': meta.zero2,
            'unknown3': meta.unknown3,
            'zero3': meta.zero3,
            'imgSizeWidth': meta.imgSizeWidth,
            'n_bscans': meta.n_bscans,
            'aktImage': meta.aktImage,
            'scan_pattern': meta.scan_pattern,
            'center_x': meta.center_x,
            'center_y': meta.center_y,
            'unknown4': meta.unknown4,
            'acquisitionTime': meta.acquisitionTime,
            'numAve': meta.numAve,
            'quality': meta.quality,
            'unknown5': meta.unknown5,
            'art_mode': meta.art_mode,
            'quality_ui': meta.quality_ui,
            'focus_candidate_raw': meta.focus_candidate_raw,
            'focus_raw': meta.focus_raw,
            'focus_diopters': meta.focus_diopters,
            'eye_length_preset_raw': meta.eye_length_preset_raw,
            'eye_length_preset': meta.eye_length_preset,
            'oct_controller_fw_version': meta.oct_controller_fw_version,
            'oct_camera_fw_version': meta.oct_camera_fw_version,
            'oct_camera_fpga_version': meta.oct_camera_fpga_version,
        }

        self._meta = EyeBscanMeta(  #quality=meta.quality,
            start_pos=((meta['start_x']),
                       (meta['start_y'])),
            end_pos=((meta['end_x']),
                     (meta['end_y'])),
            pos_unit='°',
            **meta_dict)
        return self._meta

    def get_bscan(self) -> np.ndarray:
        """Return the slice image (B-scan)"""
        bscan_folders = [
            f for f in self.folders[TypesEnum.image] if f.data.type == 35652097
        ]
        if len(bscan_folders) > 1:
            logger.warning(
                'There is more than one B-scan per slice. This is not expected.'
            )
        return bscan_folders[0].data.data

    def get_localizer(self) -> np.ndarray:
        """Return the slice image (Localizer/Fundus) For the scanpattern "OCT
        Bscan" a localizer might be stored in the E2ESliceStructure and not the
        E2ESeriesStructure."""
        localizer_folders = [
            f for f in self.folders[TypesEnum.image] if f.data.type == 33620481
        ]
        if len(localizer_folders) > 1:
            logger.warning(
                'There is more than one localizer per slice. This is not expected.'
            )
        return localizer_folders[0].data.data


class E2ESeriesStructure(E2EStructureMixin):
    """E2E Series Structure.

    This structure contains folders with data for a single Series/OCT-
    Volume and provides convenience functions for accessing the data.
    """

    def __init__(self, id: int) -> None:
        self.id = id
        self.substructure: dict[int, E2ESliceStructure] = {}
        self.folders: dict[int | str, list[E2EFolder]] = {}
        self.study: E2EStudyStructure | None = None
        self.patient: E2EPatientStructure | None = None

        self._meta = None
        self._bscan_meta = None
        self._localizer_meta = None
        self._sorted_slices_cache = None
        self._bscanmeta_items_cache = None
        self._type39_raw_cache = None
        self._section_title = ''
        self._section_description = ''

        # Description used in inspect()
        # Parts are (name, folder_id, index in list of strings)
        self._section_description_parts = [
            ('Structure:', 9005, 0),
            ('Scanpattern:', 9006, 0),
            ('Oct Modality:', 9008, 1),
            ('Enface Modality:', 9007, 1),
        ]

    def _folders_for(
        self,
        structure: E2EStructureMixin | None,
        folder_type: TypesEnum | int,
    ) -> list[E2EFolder]:
        if structure is None:
            return []
        return structure.folders.get(folder_type, [])

    def _first_folder(
        self,
        structure: E2EStructureMixin | None,
        folder_type: TypesEnum | int,
    ) -> E2EFolder | None:
        folders = self._folders_for(structure, folder_type)
        return folders[0] if folders else None

    def _first_series_folder(self, folder_type: (TypesEnum |
                                                       int)) -> E2EFolder | None:
        return self._first_folder(self, folder_type)

    def _first_study_folder(self, folder_type: (TypesEnum |
                                                      int)) -> E2EFolder | None:
        return self._first_folder(self.study, folder_type)

    def _first_patient_folder(
        self,
        folder_type: TypesEnum | int,
    ) -> E2EFolder | None:
        return self._first_folder(self.patient, folder_type)

    def _sorted_slices(self) -> list[E2ESliceStructure]:
        if self._sorted_slices_cache is not None:
            return self._sorted_slices_cache

        def sort_key(slice_structure: E2ESliceStructure) -> tuple[int, int]:
            if TypesEnum.bscanmeta not in slice_structure.folders:
                return (sys.maxsize, slice_structure.id)
            return (slice_structure._get_bscanmeta_data().aktImage,
                    slice_structure.id)

        self._sorted_slices_cache = sorted(self.slices.values(), key=sort_key)
        return self._sorted_slices_cache

    def _bscanmeta_items(self) -> list[Type10004]:
        if self._bscanmeta_items_cache is not None:
            return self._bscanmeta_items_cache

        items = []
        for slice_structure in self._sorted_slices():
            if TypesEnum.bscanmeta in slice_structure.folders:
                items.append(slice_structure._get_bscanmeta_data())
        self._bscanmeta_items_cache = items
        return self._bscanmeta_items_cache

    def _bscanmeta_item(self, bscan_index: int) -> Type10004:
        items = self._bscanmeta_items()
        if bscan_index < 0 or bscan_index >= len(items):
            raise IndexError(
                f'bscan_index {bscan_index} is out of range for {len(items)} B-scans.'
            )
        return items[bscan_index]

    def _first_type39_raw(self) -> bytes | None:
        if self._type39_raw_cache is not None:
            return self._type39_raw_cache

        for slice_structure in self._sorted_slices():
            folder = self._first_folder(slice_structure, TypesEnum.localizer_settings)
            if folder is not None:
                self._type39_raw_cache = folder.get_bytes()
                return self._type39_raw_cache
        return None

    def _type39_uint8(self, offset: int) -> int | None:
        raw = self._first_type39_raw()
        if raw is None or len(raw) <= offset:
            return None
        return raw[offset]

    def _type39_uint32(self, offset: int) -> int | None:
        raw = self._first_type39_raw()
        if raw is None or len(raw) < offset + 4:
            return None
        return struct.unpack_from('<I', raw, offset)[0]

    def _type39_version(self, offset: int) -> str | None:
        raw = self._first_type39_raw()
        if raw is None or len(raw) < offset + 4:
            return None
        return _format_version(raw[offset:offset + 4])

    def _type39_codes(self) -> list[str]:
        raw = self._first_type39_raw()
        if raw is None:
            return []
        return _extract_ascii_codes(raw)

    def _type13_strings(self) -> list[str]:
        folder = self._first_study_folder(TypesEnum.application_data)
        if folder is None:
            return []
        return folder.data.text

    def _camera_model(self) -> str | None:
        for text in self._type13_strings():
            if 'Spectralis' in text:
                return re.sub(r'\s*\+\s*', '+', text).replace('HRA+', 'HRA+')
        return None

    def _application(self) -> str | None:
        examined_structure = self._series_text(TypesEnum.examined_structure, 0)
        for text in self._type13_strings():
            if text == examined_structure:
                return text
        return examined_structure

    def _series_text(self, folder_type: TypesEnum | int,
                     index: int = 0) -> str | None:
        folder = self._first_series_folder(folder_type)
        if folder is None:
            return None
        text = folder.data.text
        if index >= len(text):
            return None
        return text[index]

    def _study_text(self, folder_type: TypesEnum | int,
                    index: int = 0) -> str | None:
        folder = self._first_study_folder(folder_type)
        if folder is None:
            return None
        text = folder.data.text
        if index >= len(text):
            return None
        return text[index]

    def _series_exam_time_iso(self) -> str | None:
        folder = self._first_series_folder(TypesEnum.examination_time)
        if folder is None:
            return None
        return folder.data.examination_time

    def _series_date_iso(self) -> str | None:
        exam_time = self._series_exam_time_iso()
        if exam_time is None:
            return None
        return datetime.datetime.fromisoformat(exam_time).date().isoformat()

    def get_focus_candidate_raw(self) -> float | None:
        items = self._bscanmeta_items()
        if not items:
            return None
        return items[0].focus_candidate_raw

    def _derive_scan_focus_diopters(self) -> float | None:
        items = self._bscanmeta_items()
        if not items:
            return None
        value = items[0].focus_diopters
        return value if math.isfinite(value) else None

    def get_progression_key(self) -> tuple[str, int] | None:
        """Current progression UUID and exact examination timestamp key."""
        folder = self._first_series_folder(10005)
        if folder is None or folder.size != 24:
            return None
        data = folder.data
        return data.progression_id, data.examination_time_raw

    def get_registration_metadata(self) -> dict[str, Any] | None:
        """Decode the observed version-3 Type 10025 reference/affine layout.

        The high flag bit is retained without assuming it means follow-
        up. Unsupported layouts remain explicit rather than yielding a
        guessed key.
        """
        folder = self._first_series_folder(10025)
        if folder is None:
            return None
        raw = folder.get_bytes()
        if len(raw) != 100:
            return {'status': 'unsupported', 'payload_bytes': len(raw)}
        version, flags = struct.unpack_from('<2I', raw)
        if version != 3 or flags & 0x7fffffff != len(raw):
            return {'status': 'unsupported', 'version': version, 'size_flags': flags}
        data = folder.data
        return {'status': 'decoded', 'version': version,
                'size_flags': flags, 'high_bit': bool(flags & 0x80000000),
                'progression_id': data.progression_id,
                'reference_time_raw': data.reference_time_raw,
                'affine': list(data.transform),
                'translation_unit': 'centered localizer pixels',
                'source': 'Type10025@series:UUID+8,time+24,affine+32'}

    def get_reference_key(self) -> tuple[str, int] | None:
        """Return the recorded reference key, without interpreting Type 6
        order."""
        registration = self.get_registration_metadata()
        if registration is None or registration['status'] != 'decoded':
            return None
        return registration['progression_id'], registration['reference_time_raw']

    def _calibration_source(self, reference_series=None):
        registration = self.get_registration_metadata()
        if registration is None:
            if reference_series is not None and reference_series is not self:
                raise ValueError('Series has no recorded reference key to validate')
            return self, 'not_recorded'
        if registration['status'] != 'decoded':
            return None, 'unsupported_registration'
        reference_key = self.get_reference_key()
        if reference_series is not None:
            if reference_series.get_progression_key() != reference_key:
                raise ValueError('Supplied reference does not match the recorded progression/time key')
            return reference_series, 'resolved'
        if reference_key == self.get_progression_key():
            return self, 'self_reference'
        folder = self._first_series_folder(10005)
        if folder is not None:
            resolved = folder.reader.resolve_reference(self)
            if resolved is not None:
                return resolved, 'resolved'
        return None, 'unresolved'

    def get_corneal_radius_metadata(self) -> dict[str, Any]:
        """Prefer acquisition Type 39 radius; fall back to the matching study
        eye."""
        radii = []
        for sl in self._sorted_slices():
            for folder in sl.folders.get(39, []):
                value = folder.data.corneal_radius_mm
                if value is not None:
                    radii.append(value)
        study_values = []
        laterality = self.laterality() if self.folders.get(TypesEnum.laterality) else None
        for folder in self._folders_for(self.study, 7):
            record = folder.data
            if str(record.eye_side) == laterality:
                value = record.c_curve_mm
                if math.isfinite(value) and value > 0:
                    study_values.append(value)
        values = radii or study_values
        unique = set(values)
        return {'corneal_radius_mm': values[0] if len(unique) == 1 else None,
                'source': ('Type39@bscan+offset457:float64' if radii else
                           'Type7@study+offset17:float64' if study_values else None),
                'constant_across_records': len(unique) == 1,
                'study_radius_mm': study_values[0] if len(set(study_values)) == 1 else None,
                'acquisition_study_conflict': bool(radii and study_values and set(radii) != set(study_values))}

    def get_reference_angles(self, localizer_angle_deg: float = 30) -> np.ndarray:
        """Return endpoints in the recorded reference frame, still in degrees.

        The default 30-degree localizer field is an assumption retained
        from the original reader. Supply the actual field for other
        acquisitions.
        """
        angles = np.array([((m.start_x, m.start_y), (m.end_x, m.end_y))
                           for m in self._bscanmeta_items()], dtype=float).reshape(-1, 2, 2)
        registration = self.get_registration_metadata()
        if registration is None:
            return angles
        if registration['status'] != 'decoded':
            raise ValueError('Unsupported E2E localizer registration layout')
        localizer = self._first_series_folder(TypesEnum.image)
        if localizer is None:
            raise ValueError('A localizer width is required to undo registration')
        return inverse_localizer_registration(angles, registration['affine'],
                                               localizer_angle_deg, localizer.data.width)

    def get_calibration_metadata(self, optical_model: E2EOpticalModel | None = None,
                                 reference_series=None,
                                 localizer_angle_deg: float = 30) -> dict[str, Any]:
        """Return decoded inputs and optional, explicitly approximate scaling.

        A follow-up with an unavailable reference never uses its current
        focus as a silent substitute. Models and the localizer field
        assumption are retained in provenance. Preset codes are not
        measured axial lengths.
        """
        if not math.isfinite(localizer_angle_deg) or localizer_angle_deg <= 0:
            raise ValueError('Localizer angular field must be finite and positive')
        items = self._bscanmeta_items()
        current = {'focus_raw': self.get_focus_candidate_raw(),
                   'focus_diopters': self._derive_scan_focus_diopters(),
                   'focus_source': 'Type10004@bscan0+offset136:float64',
                   'focus_conversion': 'raw * float32(1.043) / 1000',
                   'focus_constant_across_bscans': bool(items) and len({m.focus_raw for m in items}) == 1,
                   'eye_length_preset_raw': items[0].eye_length_preset_raw if items else None,
                   'eye_length_preset': items[0].eye_length_preset if items else None,
                   **self.get_corneal_radius_metadata()}
        reference, status = self._calibration_source(reference_series)
        effective = None
        if reference is not None and reference._bscanmeta_items():
            effective = {'focus_diopters': reference._derive_scan_focus_diopters(),
                         **reference.get_corneal_radius_metadata()}
        result = {'status': 'inputs_only', 'current_inputs': current,
                  'effective_inputs': effective, 'reference_status': status,
                  'progression_key': self.get_progression_key(),
                  'reference_key': self.get_reference_key(),
                  'registration': self.get_registration_metadata(),
                  'localizer_angle_deg': localizer_angle_deg,
                  'localizer_angle_source': 'caller supplied or default 30-degree assumption',
                  'optical_model': optical_model.as_dict() if optical_model is not None else None,
                  'mm_per_degree': None, 'scale_x_mm_per_pixel': None,
                  'source': 'full stored width and reference-frame angular span'}
        if optical_model is None or not items:
            return result
        if effective is None or effective['focus_diopters'] is None or effective['corneal_radius_mm'] is None:
            result['status'] = 'unavailable_calibration_inputs'
            return result
        if len({m.focus_raw for m in reference._bscanmeta_items()}) != 1:
            result['status'] = 'nonuniform_reference_focus'
            return result
        if any(m.scan_pattern in (2, 5) for m in items):
            raise ValueError('Approximate metric calibration requires linear or raster scans')
        angles = self.get_reference_angles(localizer_angle_deg)
        widths = np.array([m.size_x for m in items])
        if np.any(widths <= 0):
            raise ValueError('Full stored image width must be positive')
        scales_deg = np.linalg.norm(angles[:, 1] - angles[:, 0], axis=1) / widths
        if not np.isfinite(scales_deg).all() or np.any(scales_deg <= 0):
            raise ValueError('Invalid angular span or full stored image width')
        if not np.allclose(scales_deg, scales_deg[0], rtol=1e-6, atol=0):
            raise ValueError('Nonuniform lateral sampling cannot use one EyeVolume scale')
        coefficient = optical_model.mm_per_degree(effective['corneal_radius_mm'], effective['focus_diopters'])
        result.update(status='approximate', mm_per_degree=coefficient,
                      scale_x_mm_per_pixel=float(scales_deg[0] * coefficient))
        return result

    def get_x_scale_derivation(self, optical_model: E2EOpticalModel | None = None,
                               reference_series=None,
                               localizer_angle_deg: float = 30) -> dict[str, Any]:
        items = self._bscanmeta_items()
        if not items:
            return {
                'angular_width_deg': None,
                'img_size_width': None,
                'constant_um_per_deg': None,
                'scaling_x_um_per_pixel': None,
            }

        bscan_meta = items[0]
        angular_width_deg = math.hypot(bscan_meta.end_x - bscan_meta.start_x,
                                      bscan_meta.end_y - bscan_meta.start_y)
        calibration = self.get_calibration_metadata(optical_model, reference_series, localizer_angle_deg)
        scale = calibration['scale_x_mm_per_pixel']
        coefficient = calibration['mm_per_degree']
        return {
            'angular_width_deg': angular_width_deg,
            'img_size_width': bscan_meta.imgSizeWidth,
            'full_image_width': bscan_meta.size_x,
            'constant_um_per_deg': None if coefficient is None else coefficient * 1000,
            'scaling_x_um_per_pixel': None if scale is None else scale * 1000,
            'calibration': calibration,
        }

    def _oct_quality_db(self, bscan_index: int | None) -> int | None:
        if bscan_index is None:
            return None
        return round(self._bscanmeta_item(bscan_index).quality_ui)

    def _oct_art_mode(self, bscan_index: int | None) -> int | None:
        if bscan_index is None:
            return None
        return self._bscanmeta_item(bscan_index).art_mode

    def _oct_acquisition_time(self,
                              bscan_index: int | None) -> str | None:
        if bscan_index is None:
            return None
        return self._bscanmeta_item(bscan_index).acquisitionTime

    def _bscan_vertical_span_deg(self) -> float | None:
        items = self._bscanmeta_items()
        if not items:
            return None
        centers = [item.center_y for item in items]
        return max(centers) - min(centers)

    def _build_heyex_metadata(
        self,
        bscan_index: int | None = None,
        optical_model: E2EOpticalModel | None = None,
        reference_series=None,
        localizer_angle_deg: float = 30,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        patient_folder = self._first_patient_folder(TypesEnum.patient)
        patient_data = patient_folder.data if patient_folder is not None else None
        exam_time_iso = self._series_exam_time_iso()
        exam_time = _format_time(exam_time_iso)
        focus_diopters = self._derive_scan_focus_diopters()
        focus_text = _format_scan_focus(focus_diopters)
        x_scale_info = self.get_x_scale_derivation(optical_model, reference_series, localizer_angle_deg)
        scaling_x = x_scale_info['scaling_x_um_per_pixel']
        bscan_meta = self._bscanmeta_items()[0] if self._bscanmeta_items() else None
        localizer_folder = self._first_series_folder(TypesEnum.image)
        localizer_size_x = None
        localizer_size_y = None
        if localizer_folder is not None and localizer_folder.data.type == 33620481:
            localizer_size_x = localizer_folder.data.width
            localizer_size_y = localizer_folder.data.height

        oct_scan_angle = None
        size_x = None
        size_z = None
        scaling_z = None
        pattern_height_deg = self._bscan_vertical_span_deg()
        if bscan_meta is not None:
            oct_scan_angle = math.hypot(bscan_meta.end_x - bscan_meta.start_x,
                                        bscan_meta.end_y - bscan_meta.start_y)
            size_x = bscan_meta.size_x
            size_z = bscan_meta.size_y
            scaling_z = bscan_meta.scale_y * 1000

        coefficient = x_scale_info['constant_um_per_deg']
        if coefficient is not None:
            endpoints = self.get_reference_angles(localizer_angle_deg)
            oct_scan_angle = float(np.linalg.norm(endpoints[0, 1] - endpoints[0, 0]))
            centers = endpoints.mean(axis=1)
            pattern_height_deg = float(np.linalg.norm(centers[-1] - centers[0]))
        pattern_height_mm = (None if pattern_height_deg is None or coefficient is None else
                             pattern_height_deg * coefficient / 1000)
        distance_between_bscans_um = None
        if pattern_height_deg is not None and coefficient is not None and self.n_bscans > 1:
            distance_between_bscans_um = (
                pattern_height_deg * coefficient /
                (self.n_bscans - 1))
        ir_scale = (None if coefficient is None or localizer_size_x is None else
                    coefficient * localizer_angle_deg / localizer_size_x)

        model_codes = self._type39_codes()
        camera_model_code = None
        if len(model_codes) >= 2:
            camera_model_code = '/'.join(model_codes[:2])
        elif model_codes:
            camera_model_code = model_codes[0]

        ir_art_mode = self._type39_uint8(376)
        ir_dc_sensitivity = self._type39_uint8(8)
        ir_total_sensitivity = self._type39_uint8(116)
        oct_art_mode = self._oct_art_mode(bscan_index)
        oct_quality_db = self._oct_quality_db(bscan_index)
        oct_acquisition_time = _format_time(self._oct_acquisition_time(bscan_index))

        metadata = {
            'Patient': {
                'First Name': patient_data.firstname.strip()
                if patient_data is not None else None,
                'Surname': patient_data.surname.strip()
                if patient_data is not None else None,
                'Patient ID': patient_data.patient_id.strip()
                if patient_data is not None else None,
                'Date of Birth': None,
                'Sex': patient_data.sex.strip() if patient_data is not None else None,
            },
            'Container': {
                'Series ID': str(self.id),
                'Series Date': self._series_date_iso(),
                'Image Count': str(self.n_bscans),
                'Laterality': self.laterality(),
                'Scan Pattern': self._series_text(TypesEnum.scanpattern, 0),
                'Enface Modality': self.enface_modality(),
                'OCT Modality': self._series_text(TypesEnum.oct_modality, 1),
            },
            'General Parameters': {
                'Resolution Mode': None,
                'Scan Focus': focus_text,
                'Camera Objective': None,
                'Internal Target': None,
                'External Target': None,
                'Examination Time': exam_time,
                'Examined Structure': self._series_text(TypesEnum.examined_structure,
                                                        0),
                'Application': self._application(),
            },
            'IR Image': {
                'Scan Angle': f'{localizer_angle_deg:g}°' if localizer_size_x is not None else None,
                'Size X': _format_px_mm(localizer_size_x, ir_scale)
                if localizer_size_x is not None else None,
                'Size Y': _format_px_mm(localizer_size_y, ir_scale)
                if localizer_size_y is not None else None,
                'Scaling': None if ir_scale is None else f'{ir_scale:.2f} µm/pixel',
                'ART Mode': None if ir_art_mode is None else
                f'ON ({ir_art_mode} images averaged)',
                'ART Normalization': None,
                'Sensitivity (DC/DC)': None if ir_dc_sensitivity is None else
                f'{ir_dc_sensitivity}%',
                'Total Sensitivity': None if ir_total_sensitivity is None else
                str(ir_total_sensitivity),
                'IR Laser Power': None,
                'Filter State': None,
                'Lookup Table': None,
                'ERG Mode': None,
                'Auto-Brightness State': None,
                'Grey Value Offset': None,
            },
            'OCT Image': {
                'Scan Angle': None if oct_scan_angle is None else
                f'{round(oct_scan_angle):d}°',
                'Size X': _format_px_mm(size_x, scaling_x)
                if size_x is not None else None,
                'Size Z': _format_px_mm(size_z, scaling_z)
                if size_z is not None and scaling_z is not None else None,
                'Scaling X': None if scaling_x is None else
                f'{scaling_x:.2f} µm/pixel',
                'Scaling Z': None if scaling_z is None else
                f'{scaling_z:.2f} µm/pixel',
                'ART Mode': None if oct_art_mode is None else
                f'ON ({oct_art_mode} images averaged)',
                'A-Scan Rate': None,
                'Eye Length': bscan_meta.eye_length_preset if bscan_meta is not None else None,
                'Quality': None if oct_quality_db is None else f'{oct_quality_db} dB',
                'EDI Mode': None,
                'EVI Mode': None,
                'Acquisition Time': oct_acquisition_time,
            },
            'OCT Scan Pattern': {
                'Number of B-Scans': str(self.n_bscans),
                'Pattern Size': None if oct_scan_angle is None or
                pattern_height_deg is None or scaling_x is None or
                pattern_height_mm is None else
                f'{round(oct_scan_angle):d}° x {pattern_height_deg:.1f}° ({size_x * scaling_x / 1000:.1f} x {pattern_height_mm:.1f} mm)',
                'Distance between B-Scans': None if distance_between_bscans_um is None
                else f'{round(distance_between_bscans_um):d} µm',
            },
            'Device': {
                'Camera Model': self._camera_model(),
                'Camera Model Code': camera_model_code,
                'Camera S/N': None if self._type39_uint32(104) is None else
                f'{self._type39_uint32(104):06d}',
                'Power Supply S/N': None if self._type39_uint32(108) is None else
                f'{self._type39_uint32(108):06d}',
                'Touch Panel S/N': None if self._type39_uint32(112) is None else
                f'{self._type39_uint32(112):06d}',
                'HRA Camera FW Version': self._type39_version(128),
                'Power Supply FW Version': self._type39_version(132),
                'Touch Panel FW Version': self._type39_version(136),
                'OCT Camera FW Version': None if bscan_meta is None else
                bscan_meta.oct_camera_fw_version,
                'OCT Controller FW Version': None if bscan_meta is None else
                bscan_meta.oct_controller_fw_version,
                'OCT Camera FPGA Version': None if bscan_meta is None else
                bscan_meta.oct_camera_fpga_version,
                'Acquisition Software Version': self._type39_version(140),
            },
        }

        sources = {
            'Patient': {
                'First Name': 'Type9@patient+offset0:ascii'
                if patient_data is not None else None,
                'Surname': 'Type9@patient+offset31:ascii'
                if patient_data is not None else None,
                'Patient ID': 'Type9@patient+offset106:ascii'
                if patient_data is not None else None,
                'Date of Birth': None,
                'Sex': 'Type9@patient+offset101:ascii'
                if patient_data is not None else None,
            },
            'Container': {
                'Series ID': 'ContainerHeader.series_id',
                'Series Date': 'derived from Type10005@series+offset16',
                'Image Count': 'derived from Type10004.n_bscans',
                'Laterality': 'Type11@series+offset14',
                'Scan Pattern': 'Type9006@series[0]',
                'Enface Modality': 'Type9007@series[1]',
                'OCT Modality': 'Type9008@series[1]',
            },
            'General Parameters': {
                'Resolution Mode': None,
                'Scan Focus': 'empirical conversion of Type10004@bscan0+offset136:float64',
                'Camera Objective': None,
                'Internal Target': None,
                'External Target': None,
                'Examination Time': 'Type10005@series+offset16:DateTime',
                'Examined Structure': 'Type9005@series[0]',
                'Application': 'Type13@study (fallback Type9005@series[0])',
            },
            'IR Image': {
                'Scan Angle': 'derived from current localizer field size assumption',
                'Size X': 'derived from localizer width and provisional Scaling',
                'Size Y': 'derived from localizer height and provisional Scaling',
                'Scaling': 'explicit optical model and declared localizer field; unavailable by default',
                'ART Mode': 'Type39@slice0+offset376:uint8',
                'ART Normalization': None,
                'Sensitivity (DC/DC)': 'Type39@slice0+offset8:uint8',
                'Total Sensitivity': 'Type39@slice0+offset116:uint8',
                'IR Laser Power': None,
                'Filter State': None,
                'Lookup Table': None,
                'ERG Mode': None,
                'Auto-Brightness State': None,
                'Grey Value Offset': None,
            },
            'OCT Image': {
                'Scan Angle': 'derived from Type10004.start_x/end_x',
                'Size X': 'Type10004.size_x plus provisional Scaling X',
                'Size Z': 'Type10004.size_y and Type10004.scale_y',
                'Scaling X': 'explicit optical model, reference-frame span and full Type10004.size_x; unavailable by default',
                'Scaling Z': 'Type10004.scale_y',
                'ART Mode': None if bscan_index is None else
                f'Type10004@bscan{bscan_index}+offset120:uint32',
                'A-Scan Rate': None,
                'Eye Length': 'Type10004@bscan0+offset144:float64 preset code, not measured axial length',
                'Quality': None if bscan_index is None else
                f'Type10004@bscan{bscan_index}+offset156:float32',
                'EDI Mode': None,
                'EVI Mode': None,
                'Acquisition Time': None if bscan_index is None else
                f'Type10004@bscan{bscan_index}+offset88:DateTime',
            },
            'OCT Scan Pattern': {
                'Number of B-Scans': 'Type10004.n_bscans',
                'Pattern Size': 'derived from Type10004 centers and provisional Scaling X',
                'Distance between B-Scans': 'reference-frame centers and explicit optical model; unavailable by default',
            },
            'Device': {
                'Camera Model': 'Type13@study',
                'Camera Model Code': 'Type39 raw ASCII model codes',
                'Camera S/N': 'Type39@slice0+offset104:uint32',
                'Power Supply S/N': 'Type39@slice0+offset108:uint32',
                'Touch Panel S/N': 'Type39@slice0+offset112:uint32',
                'HRA Camera FW Version': 'Type39@slice0+offset128:bytes4',
                'Power Supply FW Version': 'Type39@slice0+offset132:bytes4',
                'Touch Panel FW Version': 'Type39@slice0+offset136:bytes4',
                'OCT Camera FW Version': 'Type10004@bscan0+offset128:bytes4 (may need swapping with FPGA later)',
                'OCT Controller FW Version': 'Type10004@bscan0+offset124:bytes4',
                'OCT Camera FPGA Version': 'Type10004@bscan0+offset132:bytes4 (may need swapping with camera FW later)',
                'Acquisition Software Version': 'Type39@slice0+offset140:bytes4',
            },
        }

        return metadata, sources

    def get_heyex_metadata(
        self,
        bscan_index: int | None = None,
        **calibration_options,
    ) -> dict[str, dict[str, Any]]:
        metadata, _ = self._build_heyex_metadata(bscan_index=bscan_index, **calibration_options)
        return metadata

    def get_heyex_metadata_sources(
        self,
        bscan_index: int | None = None,
        **calibration_options,
    ) -> dict[str, dict[str, Any]]:
        _, sources = self._build_heyex_metadata(bscan_index=bscan_index, **calibration_options)
        return sources

    def get_eyevolume_e2e_metadata(self, **calibration_options) -> dict[str, Any]:
        """Return a compact E2E metadata bundle suitable for EyeVolumeMeta."""
        return {
            'heyex': self.get_heyex_metadata(**calibration_options),
            'heyex_sources': self.get_heyex_metadata_sources(**calibration_options),
            'heyex_bscan_oct': [
                self.get_heyex_metadata(bscan_index=i, **calibration_options)['OCT Image']
                for i in range(self.n_bscans)
            ],
            'heyex_bscan_oct_sources': [
                self.get_heyex_metadata_sources(bscan_index=i, **calibration_options)['OCT Image']
                for i in range(self.n_bscans)
            ],
            'findings': {
                'scan_focus': {
                    'raw_candidate': self.get_focus_candidate_raw(),
                    'derived_diopters': self._derive_scan_focus_diopters(),
                    'source': 'Type10004@bscan0+offset136:float64',
                    'status': 'empirically verified for supplied acquisitions',
                    'formula': 'focus_D = raw * float32(1.043) / 1000',
                },
                'scaling_x': {
                    **self.get_x_scale_derivation(**calibration_options),
                    'source': 'reference-frame angular span / full Type10004.size_x',
                    'status': 'requires explicit approximate optical model',
                },
                'calibration': self.get_calibration_metadata(**calibration_options),
                'oct_firmware_assignment': {
                    'oct_controller_fw_offset': 124,
                    'oct_camera_fw_offset': 128,
                    'oct_camera_fpga_offset': 132,
                    'status': 'provisional',
                    'note':
                    'OCT Camera FW and OCT Camera FPGA may need to be swapped later if a dataset with distinct values proves the order is reversed.',
                },
                'type39_storage': {
                    'status': 'observed',
                    'note':
                    'In the current sample set Type39 is stored in every B-scan slice and is volume-constant within a series.',
                },
            },
        }

    def add_folder(self, folder: E2EFolder) -> None:
        """Add a folder to the Series.

        Args:
            folder: The folder to add.
        """
        if folder.slice_id == -1:
            try:
                self.folders[folder.type].append(folder)
            except KeyError:
                self.folders[folder.type] = [folder]
        else:
            if folder.slice_id not in self.slices:
                self.slices[folder.slice_id] = E2ESliceStructure(
                    folder.slice_id)
            self.slices[folder.slice_id].add_folder(folder)

    def inspect(self,
                recursive: bool = False,
                ind_prefix: str = '',
                tables: bool = False) -> str:
        """Inspect the series.

        Custom `inspect` method to print a summary table for the slices belonging to the series.

        Args:
            recursive: If True inspect lower level structures recursively.
            ind_prefix: Indentation for showing information from lower level structures.
            tables: If True add markdown table overview of the contained folder types.
        """
        laterality = self.folders[TypesEnum.laterality][
            0].data.laterality.name if TypesEnum.laterality in self.folders else 'Unknown'
        text = self._get_section_title(
        ) + f' - Laterality: {laterality} - B-scans: {self.n_bscans}\n'
        text += self._get_section_description() + '\n'
        if tables:
            text += self._get_folder_summary() + '\n'

        if not recursive:
            return text

        # Describe all slices in one table
        s_data = defaultdict(list)
        for sl in self.slices.values():
            for f_list in sl.folders.values():
                for f in f_list:
                    s_data[f.type].append(f.size)

        if len(s_data) == 0 or tables == False:
            text += ''
        else:
            text += '\nE2ESlice Summary:\n'
            text += indent(self._get_table(s_data, 'E2ESliceStructure'),
                           ind_prefix)
            text += '\n'
        return text

    def get_volume(self, *, optical_model: E2EOpticalModel | None = None,
                   reference_series: E2ESeriesStructure | None = None,
                   localizer_angle_deg: float = 30) -> EyeVolume:
        """Return an EyeVolume, optionally using explicit approximate calibration.

        Args:
            optical_model: None keeps pixel units; a model opts into approximate mm.
            reference_series: External reference whose exact recorded key must match.
            localizer_angle_deg: Declared square localizer field for calibration;
                30 degrees is an assumption, not a decoded universal field.
        """
        ## Check if scan is a volume scan
        volume_meta = self.get_meta()

        scan_pattern = volume_meta['bscan_meta'][0]['scan_pattern']

        ## Check if scan pattern is supported by EyeVolume
        if scan_pattern == 2:
            msg = f'The EyeVolume object does not support scan pattern 2 (one Circular B-scan).'
            raise ValueError(msg)
        elif scan_pattern == 5:
            msg = f'The EyeVolume object does not support scan pattern 5 (Radial scan - star pattern).'
            raise ValueError(msg)

        if optical_model is not None:
            options = dict(optical_model=optical_model, reference_series=reference_series,
                           localizer_angle_deg=localizer_angle_deg)
            calibration = self.get_calibration_metadata(**options)
            if calibration['status'] != 'approximate':
                raise ValueError(f"Metric calibration unavailable: {calibration['reference_status']}; {calibration['status']}")

        data = self.get_bscans()
        volume_meta = deepcopy(self.get_meta())
        localizer = self.get_localizer()
        if optical_model is not None:
            coefficient = calibration['mm_per_degree']
            calibration_source, _ = self._calibration_source(reference_series)
            # Reference-frame endpoints must be paired with the reference
            # localizer, not overlaid on an unwarped follow-up image.
            localizer = calibration_source.get_localizer()
            angles = self.get_reference_angles(localizer_angle_deg)
            centers = angles.mean(axis=1)
            spacings = np.linalg.norm(np.diff(centers, axis=0), axis=1) * coefficient
            if len(spacings) and not np.allclose(spacings, spacings[0], rtol=1e-5, atol=1e-9):
                raise ValueError('Nonuniform raster spacing cannot use one EyeVolume scale')
            axial = [m.scale_y for m in self._bscanmeta_items()]
            if not np.isfinite(axial).all() or np.any(np.asarray(axial) <= 0) or not np.allclose(axial, axial[0]):
                raise ValueError('Invalid or nonuniform axial sampling')
            volume_meta['scale_x'] = calibration['scale_x_mm_per_pixel']
            volume_meta['scale_y'] = axial[0]
            volume_meta['scale_z'] = float(spacings.mean()) if len(spacings) else 0
            volume_meta['scale_unit'] = 'mm'
            for meta, endpoints in zip(volume_meta['bscan_meta'], angles):
                start, end = (endpoints + localizer_angle_deg / 2) * coefficient
                meta['start_pos'], meta['end_pos'] = tuple(start), tuple(end)
                meta['pos_unit'] = 'mm'
            localizer = EyeEnface(localizer.data, deepcopy(localizer.meta))
            localizer.meta['scale_x'] = coefficient * localizer_angle_deg / localizer.size_x
            localizer.meta['scale_y'] = coefficient * localizer_angle_deg / localizer.size_y
            localizer.meta['scale_unit'] = 'mm'
            volume_meta['e2e_metadata'] = self.get_eyevolume_e2e_metadata(**options)
            volume_meta['e2e_metadata']['findings']['calibration']['localizer_frame'] = 'effective reference'
        volume = EyeVolume(
            data=data,
            meta=volume_meta,
            localizer=localizer,
            transformation=_compute_localizer_oct_transform(
                volume_meta, localizer.meta, data.shape),
        )

        layer_height_maps = self.get_layers()
        layer_manual_maps = self.get_layer_manual_maps()
        for name, i in SEG_MAPPING.items():
            if i in layer_height_maps:
                volume.add_layer_annotation(
                    layer_height_maps[i],
                    name=name,
                    manual=layer_manual_maps.get(i),
                )

        return volume

    @property
    def n_bscans(self) -> int:
        """Return the number of B-scans in the series."""
        return len(self._bscanmeta_items())

    def get_bscans(self) -> np.ndarray:
        volume_meta = self.get_meta()
        size_x = volume_meta['bscan_meta'][0]['size_x']
        size_y = volume_meta['bscan_meta'][0]['size_y']

        data = np.zeros((self.n_bscans, size_y, size_x), dtype=np.float32)
        for i, sl in enumerate(s for s in self._sorted_slices() if TypesEnum.bscanmeta in s.folders):
            bscan = sl.get_bscan()
            data[i] = bscan

        return data

    def get_layers(self) -> dict[int, np.ndarray]:
        """Return layer height maps for the series as dict of numpy arrays where
        the key is the layer id."""
        slice_layers = {}
        layer_ids = set()

        for ind, sl in enumerate(s for s in self._sorted_slices() if TypesEnum.bscanmeta in s.folders):
            layers = sl.get_layers()
            [layer_ids.add(k) for k in layers.keys()]
            slice_layers[ind] = layers

        layers = {}
        size_x = self.get_bscan_meta()[0]['size_x']
        for i in layer_ids:
            layer = np.full((self.n_bscans, size_x), np.nan)
            for sl in range(self.n_bscans):
                if i in slice_layers[sl]:
                    layer[sl, :] = slice_layers[sl][i]

            layer[layer >= 3.0e+38] = np.nan
            layers[i] = layer

        return layers

    def get_layer_manual_maps(self) -> dict[int, list[bool | None]]:
        """Return manual edit flags for each layer and B-scan.

        Entries may be ``None`` for B-scans without a corresponding
        layer annotation.
        """
        slice_layer_manual = {}
        layer_ids = set()

        for ind, sl in enumerate(s for s in self._sorted_slices() if TypesEnum.bscanmeta in s.folders):
            manual_flags = sl.get_layer_manual_flags()
            layer_ids.update(manual_flags.keys())
            slice_layer_manual[ind] = manual_flags

        manual_maps = {}
        for i in layer_ids:
            manual = [None] * self.n_bscans
            for sl in range(self.n_bscans):
                manual[sl] = slice_layer_manual[sl].get(i)

            manual_maps[i] = manual

        return manual_maps

    def enface_modality(self) -> str:
        folders = self.folders[TypesEnum.enface_modality]
        if len(folders) > 1:
            logger.debug(
                'There is more than one enface modality stored. This is not expected.'
            )
        text = folders[0].data.text[1]
        return 'NIR' if text == 'IR' else text

    def laterality(self) -> str:
        folders = self.folders[TypesEnum.laterality]
        if len(folders) > 1:
            logger.debug(
                'There is more than one laterality stored. This is not expected.'
            )
        return str(folders[0].data.laterality)

    def slo_data(self) -> Type10025:
        folders = self.folders[TypesEnum.slodata]
        if len(folders) > 1:
            logger.debug(
                'There is more than one SLO data folder. This is not expected.'
            )
        return folders[0].data

    def localizer_meta(self, height, width) -> EyeEnfaceMeta:
        """Return EyeEnfaceMeta object for the localizer image."""
        if self._localizer_meta is None:
            self._localizer_meta = EyeEnfaceMeta(
                scale_x=30 / width,  # Give scale in degrees per pixel
                scale_y=30 / height,
                scale_unit='°',
                modality=self.enface_modality(),
                laterality=self.laterality(),
                field_size=None,
                scan_focus=None,
                visit_date=None,
                exam_time=None,
            )
        logger.info(
            'The localizer scale is currently hardcoded and not read from the E2E file. If you know how or where to find the scale information let us know by opening an issue.'
        )
        return self._localizer_meta

    def get_localizer(self) -> EyeEnface:
        """Return EyeEnface object for the localizer image."""
        try:
            folders = self.folders[TypesEnum.image]
            if len(folders) > 1:
                logger.warning(
                    'There is more than one enface localizer image stored. This is not expected.'
                )

            # Slodata is not always present in E2E files.
            # Todo: Give transform to EyeEnface object where it is applied to the image. EyeEnface then by default has an identity transform.
            #transform = np.array(list(self.slo_data().transform) +
            #                     [0, 0, 1]).reshape((3, 3))
            # transfrom localizer with transform from E2E file
            #transformed_localizer = warp(folders[0].data.data,
            #                             AffineTransform(transform),
            #                             order=1,
            #                             preserve_range=True)
            return EyeEnface(folders[0].data.data,
                             self.localizer_meta(height=folders[0].data.height,
                                                 width=folders[0].data.width))
        except KeyError:
            if self.n_bscans == 1:
                slice_struct = self.slices[2]
                return EyeEnface(slice_struct.get_localizer(),
                                 self.localizer_meta())
            else:
                raise ValueError(
                    'There is no localizer/fundus image in the E2E file.')

    def get_bscan_meta(self) -> list[EyeBscanMeta]:
        """Return EyeBscanMeta objects for all B-scans in the series."""
        if self._bscan_meta is None:
            self._bscan_meta = [
                sl.get_meta() for sl in self._sorted_slices()
                if TypesEnum.bscanmeta in sl.folders
            ]
        return self._bscan_meta

    def get_meta(self) -> EyeVolumeMeta:
        """Return EyeVolumeMeta object for the series."""
        if self._meta is None:
            bscan_meta = self.get_bscan_meta()
            self._meta = EyeVolumeMeta(
                scale_x=1,  #0.0114,  # Todo: Where is this in E2E?
                scale_y=1,  #bscan_meta[0]["scale_y"],
                scale_z=1,  #get_bscan_spacing(bscan_meta) if
                #(bscan_meta[0]["scan_pattern"] not in [1, 2]) else 0.03,
                scale_unit='px',
                laterality=self.laterality(),
                visit_date=None,
                exam_time=None,
                bscan_meta=bscan_meta,
                intensity_transform='e2e',
                e2e_metadata=self.get_eyevolume_e2e_metadata(),
            )
        return self._meta

    @property
    def slices(self) -> dict[int, E2ESliceStructure]:
        """Alias for substructure."""
        return self.substructure


class E2EStudyStructure(E2EStructureMixin):
    """E2E Study Structure."""

    def __init__(self, id) -> None:
        self.id = id
        self.substructure: dict[int, E2ESeriesStructure] = {}
        self.folders: dict[int | str, list[E2EFolder]] = {}
        self.patient: E2EPatientStructure | None = None

        self._section_description_parts = [('Device:', 9001, 0),
                                           ('Studyname:', 9000, 0)]
        self._section_title = ''
        self._section_description = ''

    @property
    def series(self) -> dict[int, E2ESeriesStructure]:
        return self.substructure

    def add_folder(self, folder: E2EFolder) -> None:
        """Add a folder to the Study.

        Args:
            folder: The folder to add.
        """
        if folder.series_id == -1:
            try:
                self.folders[folder.type].append(folder)
            except KeyError:
                self.folders[folder.type] = [folder]
        else:
            if folder.series_id not in self.series:
                self.series[folder.series_id] = E2ESeriesStructure(
                    folder.series_id)
                self.series[folder.series_id].study = self
                self.series[folder.series_id].patient = getattr(
                    self, 'patient', None)
            self.series[folder.series_id].add_folder(folder)


class E2EPatientStructure(E2EStructureMixin):
    """E2E Patient Structure."""

    def __init__(self, id) -> None:
        self.id = id
        self.substructure: dict[int, E2EStudyStructure] = {}
        self.folders: dict[int | str, list[E2EFolder]] = {}

        self._section_description_parts = []
        self._section_title = ''
        self._section_description = ''

    @property
    def studies(self) -> dict[int, E2EStudyStructure]:
        return self.substructure

    def add_folder(self, folder: E2EFolder) -> None:
        """Add a folder to the Patient Structure.

        Args:
            folder: The folder to add.
        """
        if folder.study_id == -1:
            try:
                self.folders[folder.type].append(folder)
            except KeyError:
                self.folders[folder.type] = [folder]
        else:
            if folder.study_id not in self.studies:
                self.studies[folder.study_id] = E2EStudyStructure(
                    folder.study_id)
                self.studies[folder.study_id].patient = self
            self.studies[folder.study_id].add_folder(folder)


class E2EFileStructure(E2EStructureMixin):
    """E2E File Structure."""

    def __init__(self):
        self.substructure: dict[int, E2EPatientStructure] = {}
        self.folders: dict[int | str, list[E2EFolder]] = {}

        self._section_description_parts = []
        self._section_title = ''
        self._section_description = ''

    @property
    def patients(self) -> dict[int, E2EPatientStructure]:
        return self.substructure

    def add_folder(self, folder: E2EFolder):
        """Add a folder to the File Structure.

        Args:
            folder: The folder to add.
        """
        try:
            self.all_folders.append(folder)
        except AttributeError:
            self.all_folders = [folder]

        if folder.patient_id == -1:
            try:
                self.folders[folder.type].append(folder)
            except KeyError:
                self.folders[folder.type] = [folder]
        else:
            if folder.patient_id not in self.patients:
                self.patients[folder.patient_id] = E2EPatientStructure(
                    folder.patient_id)
            self.patients[folder.patient_id].add_folder(folder)


class HeE2eReader(AbstractContextManager):

    def __init__(self, path: str | Path):
        """Index an E2E file.

        Initialization of the HeE2eReader class indexes the specified E2E file. This allows for printing the reader object
        for a quick overview of the files contents. If you want to access the data, the reader has to be used as a Context Manager.

        ```python
        with HeE2eReader("path/to/file.e2e") as reader:
            data = reader.volumes
        ```

        Args:
            path: Path to the e2e file.
        """
        self.path = Path(path)
        self.file_object: BufferedReader

        # Index file to create hierarchy
        self.file_hierarchy = E2EFileStructure()
        self._index_file()

    def _index_file(self) -> None:
        with open(self.path, 'rb') as f:
            parsed = e2e_format.parse_stream(f)

            # Get the position, IDs and types of all folders
            for chunk in parsed.chunks:
                for fh in chunk.folders:
                    folder = E2EFolder(
                        **{
                            'patient_id': fh.patient_id,
                            'study_id': fh.study_id,
                            'series_id': fh.series_id,
                            'slice_id': fh.slice_id,
                            'pos': fh.pos,
                            'start': fh.start,
                            'type': fh.type,
                            'size': fh.size,
                            'ind': fh.ind,
                            'reader': self,
                        })
                    self.file_hierarchy.add_folder(folder)

            # Read and cache information required for __str__
            self.file_object = f
            self.inspect(recursive=True, ind_prefix='  ', tables=False)

    def inspect(self,
                recursive: bool = False,
                ind_prefix: str = '',
                tables: bool = True) -> str:
        """Inspect the file hierarchy (contents) of the file.

        Args:
            recursive: If True inspect lower level structures recursively.
            ind_prefix: Indentation for showing information from lower level structures.
            tables: If True add markdown table overview of the contained folder types.
        """
        return self.file_hierarchy.inspect(recursive, ind_prefix, tables)

    def __str__(self) -> str:
        return self.inspect(recursive=True, ind_prefix='  ', tables=False)

    def __repr__(self) -> str:
        return f'HeE2eReader(path="{self.path}")'

    @property
    def patients(self) -> list[E2EPatientStructure]:
        """List of all patients in the file as E2EPatient objects."""
        return [p for p in self.file_hierarchy.patients.values()]

    @property
    def studies(self) -> list[E2EStudyStructure]:
        """List of all studies in the file as E2EStudy objects."""
        studies = []
        for p in self.patients:
            studies += p.studies.values()
        return studies

    @property
    def series(self) -> list[E2ESeriesStructure]:
        """List of all series in the file as E2ESeries objects."""
        series = []
        for s in self.studies:
            series += s.series.values()
        return sorted(series, key=lambda s: s.id)

    def __enter__(self) -> HeE2eReader:
        self.file_object = open(self.path, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.file_object.close()

    def resolve_reference(self, series: E2ESeriesStructure,
                          additional_series=()) -> E2ESeriesStructure | None:
        """Match a reference by progression UUID and exact timestamp key.

        External series must belong to readers kept open while accessing
        their payloads. Missing references return None; ambiguous
        matches raise. Type 6 membership order and Type 10025's high bit
        are not substitutes.
        """
        key = series.get_reference_key()
        if key is None:
            return None
        if series.get_progression_key() == key:
            return series
        matches = {id(candidate): candidate for candidate in (*self.series, *additional_series)
                   if candidate.get_progression_key() == key}
        if len(matches) > 1:
            raise ValueError('Ambiguous E2E progression/reference timestamp key')
        return next(iter(matches.values()), None)

    def find_int(self,
                 value: int,
                 excluded_folders: list[(int |
                                              str)] = ['images', 'layers'],
                 slice_id: int | None = None,
                 **kwargs: Any) -> dict[int, dict[int, dict[str, list[int]]]]:
        """Find an integer value in the e2e file.

        Args:
            value: The value to find.
            excluded_folders: A list of folders to exclude from the search.
                None: Exclude no folders.
                "images": Exclude image data from search.
                "layers": Exclude layer data from search.
            slice_id: The slice id to search in.
            **kwargs: Keyword arguments passed to [`find_int`][eyepy.io.utils.find_int].

        Returns:
            A dictionary of the form {series_id(int): {folder_type(int): {fmt_string(str): [positions(int)]}}}
        """
        if 'images' in excluded_folders:
            excluded_folders[excluded_folders.index('images')] = 1073741824
        if 'layers' in excluded_folders:
            excluded_folders[excluded_folders.index('layers')] = 10019

        results = defaultdict(dict)
        for folder in self.file_hierarchy.all_folders:
            if not int(folder.type) in excluded_folders and (
                    True if slice_id is None else folder.slice_id == slice_id):
                res = find_int(folder.get_bytes(), value, **kwargs)
                if res:
                    results[folder.series_id][folder.type] = res
        results = {**results}
        return results

    def find_float(self,
                   value: float,
                   excluded_folders: list[(int |
                                                str)] = ['images', 'layers'],
                   slice_id: int | None = None,
                   **kwargs: Any) -> dict[int, dict[int, dict[str, list[int]]]]:
        """Find a float value in the e2e file.

        Args:
            value: The value to find.
            excluded_folders: A list of folders to exclude from the search.
                None: Exclude no folders.
                "images": Exclude image data from search.
                "layers": Exclude layer data from search.
            slice_id: The slice to search in. Specify 0 if you do not want to search through all slices but one slice per volume is enough.
            **kwargs: Keyword arguments passed to [`find_float`][eyepy.io.utils.find_float].

        Returns:
            A dictionary of the form {series_id(int): {folder_type(int): {fmt_string(str): [positions(int)]}}}
        """
        if 'images' in excluded_folders:
            excluded_folders[excluded_folders.index('images')] = 1073741824
        if 'layers' in excluded_folders:
            excluded_folders[excluded_folders.index('layers')] = 10019

        results = defaultdict(dict)
        for folder in self.file_hierarchy.all_folders:
            if not int(folder.type) in excluded_folders and (
                    True if slice_id is None else folder.slice_id == slice_id):
                res = find_float(folder.get_bytes(), value, **kwargs)
                if res:
                    results[folder.series_id][folder.type] = res
        results = {**results}
        return results

    def find_number(self,
                    value: int | float,
                    excluded_folders: list[(int |
                                                 str)] = ['images', 'layers'],
                    slice_id: int | None = None,
                    **kwargs: Any) -> dict[int, dict[int, dict[str, list[int]]]]:
        """Find a number value in the e2e file.

        Use this function if you don't know if the value is an integer or a float.
        This is just a shortcut for calling [`find_int`][eyepy.io.he.e2e_reader.HeE2eReader.find_int]
        and [`find_float`][eyepy.io.he.e2e_reader.HeE2eReader.find_float] individually.

        Args:
            value: The value to find.
            excluded_folders: A list of folders to exclude from the search.
                None: Exclude no folders.
                "images": Exclude image data from search.
                "layers": Exclude layer data from search.
            slice_id: The slice to search in. Specify 0 if you do not want to search through all slices but one slice per volume is enough.
            **kwargs: Keyword arguments passed to [`find_int`][eyepy.io.utils.find_int] and [`find_float`][eyepy.io.utils.find_float].

        Returns:
            A dictionary of the form {series_id(int): {folder_type(int): {fmt_string(str): [positions(int)]}}}
        """
        results = {
            **self.find_float(value, excluded_folders, slice_id, **kwargs),
            **self.find_int(round(value), excluded_folders, slice_id, **kwargs)
        }
        return results

    @property
    def volume(self) -> EyeVolume:
        """First EyeVolume object in the E2E file.

        Returns:
            EyeVolume object for the first Series in the e2e file.
        """
        for s in self.series:
            try:
                return s.get_volume()
            except Exception as e:
                # for compatibility with python <= 3.9, later work with only the exception as argument for format_exception
                exc_type, exc_value, exc_tb = sys.exc_info()
                logger.debug(''.join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)))
        raise ValueError(
            'No Series in the E2E file can be parsed to a an EyeVolume object. You might be able to extract information manually from the E2ESeries objects (e2ereader.series)'
        )

    @property
    def volumes(self) -> list[EyeVolume]:
        """All EyeVolume objects in the E2E file.

        Returns:
            List with EyeVolume objects for every Series in the e2e file.
        """
        volumes = []
        for s in self.series:
            try:
                volumes.append(s.get_volume())
            except Exception as e:
                logger.debug(''.join(traceback.format_exception(e)))
        return volumes
