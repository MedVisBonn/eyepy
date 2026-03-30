from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
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

from .e2e_format import ContainerHeader
from .e2e_format import containerheader_format
from .e2e_format import DataContainer
from .e2e_format import datacontainer_format
from .e2e_format import e2e_format
from .e2e_format import Type10004
from .e2e_format import Type10025
from .e2e_format import TypesEnum
from .vol_reader import SEG_MAPPING

logger = logging.getLogger(__name__)

# Occurence of type ids. This is used by the inspect function.
type_occurence = {
    'E2EFileStructure': [0, 9011],
    'E2EPatientStructure': [9, 17, 29, 31, 52, 9010],
    'E2EStudyStructure': [7, 10, 13, 30, 53, 58, 1000, 9000, 9001],
    'E2ESeriesStructure': [
        2, 3, 11, 54, 59, 61, 62, 1000, 1001, 1003, 1008, 9005, 9006, 9007,
        9008, 10005, 10009, 10010, 10011, 10013, 10025, 1073741824, 1073751824,
        1073751825, 1073751826
    ],
    'E2ESliceStructure': [
        2, 3, 5, 39, 40, 10004, 10012, 10013, 10019, 10020, 10032, 1073741824,
        1073751824, 1073751825, 1073751826
    ]
}

HEYEX_TIMEZONE_SUFFIX = 'UTC+1'
MEDIUM_EYE_LENGTH_UM_PER_DEG = 289.6


def _format_version(raw: bytes) -> str:
    return '.'.join(str(part) for part in raw)


def _format_px_mm(size_px: int, scale_um_per_px: Optional[float]) -> str:
    if scale_um_per_px is None:
        return f'{size_px} pixels'
    return f'{size_px} pixels ({size_px * scale_um_per_px / 1000:.1f} mm)'


def _format_scan_focus(value_diopters: Optional[float]) -> Optional[str]:
    if value_diopters is None:
        return None
    return f'{value_diopters:.2f} D'


def _format_time(iso_value: Optional[str],
                 timezone_suffix: str = HEYEX_TIMEZONE_SUFFIX) -> Optional[str]:
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
        folder_type: Union[TypesEnum, int],
        offset: int = 0,
        data_construct: Optional[Union[cs.Construct, str]] = None,
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
                'pandas is required for table output. Please install pandas or use the inspect function without tables=True.')
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

    @property
    def file_object(self) -> BufferedReader:
        """Return the file object.

        This refers to the the HeE2eReader file object.
        """
        return self.reader.file_object

    @property
    def data(self) -> Any:
        """Return the data."""
        if not self._data:
            parsed = self._parse_data()
            self._data = parsed.item
            self._header = parsed.header
        return self._data

    @property
    def header(self) -> ContainerHeader:
        """Return the data header."""
        if not self._header:
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
        self.file_object.seek(self.start)
        return datacontainer_format.parse_stream(self.file_object)

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
        self.file_object.seek(self.start + containerheader_format.sizeof())
        return self.file_object.read(self.size)


class E2ESliceStructure(E2EStructureMixin):
    """E2E Slice Structure.

    This structure contains folders with data for a single Slice/B-csan
    and provide convenience functions for accessing the data.
    """

    def __init__(self, id: int) -> None:
        self.id = id
        self.folders: dict[Union[int, str], list[E2EFolder]] = {}

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
        for layer_folder in self.folders[TypesEnum.layer_annotation]:
            layers[layer_folder.data.id] = layer_folder.data.data
        return layers

    def get_meta(self) -> EyeBscanMeta:
        """Return the slice meta data."""
        if len(self.folders[TypesEnum.bscanmeta]) > 1:
            logger.warning(
                'There is more than one bscanmeta object. This is not expected.'
            )

        meta = self.folders[TypesEnum.bscanmeta][0].data
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
            'oct_controller_fw_version': meta.oct_controller_fw_version,
            'oct_camera_fw_version': meta.oct_camera_fw_version,
            'oct_camera_fpga_version': meta.oct_camera_fpga_version,
        }

        return EyeBscanMeta(  #quality=meta.quality,
            start_pos=((meta['start_x']),
                       (meta['start_y'])),
            end_pos=((meta['end_x']),
                     (meta['end_y'])),
            pos_unit='°',
            **meta_dict)

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
        self.folders: dict[Union[int, str], list[E2EFolder]] = {}
        self.study: Optional[E2EStudyStructure] = None
        self.patient: Optional[E2EPatientStructure] = None

        self._meta = None
        self._bscan_meta = None
        self._localizer_meta = None
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
        structure: Optional[E2EStructureMixin],
        folder_type: Union[TypesEnum, int],
    ) -> list[E2EFolder]:
        if structure is None:
            return []
        return structure.folders.get(folder_type, [])

    def _first_folder(
        self,
        structure: Optional[E2EStructureMixin],
        folder_type: Union[TypesEnum, int],
    ) -> Optional[E2EFolder]:
        folders = self._folders_for(structure, folder_type)
        return folders[0] if folders else None

    def _first_series_folder(self, folder_type: Union[TypesEnum,
                                                       int]) -> Optional[E2EFolder]:
        return self._first_folder(self, folder_type)

    def _first_study_folder(self, folder_type: Union[TypesEnum,
                                                      int]) -> Optional[E2EFolder]:
        return self._first_folder(self.study, folder_type)

    def _first_patient_folder(
        self,
        folder_type: Union[TypesEnum, int],
    ) -> Optional[E2EFolder]:
        return self._first_folder(self.patient, folder_type)

    def _sorted_slices(self) -> list[E2ESliceStructure]:
        def sort_key(slice_structure: E2ESliceStructure) -> tuple[int, int]:
            if TypesEnum.bscanmeta not in slice_structure.folders:
                return (sys.maxsize, slice_structure.id)
            return (slice_structure.folders[TypesEnum.bscanmeta][0].data.aktImage,
                    slice_structure.id)

        return sorted(self.slices.values(), key=sort_key)

    def _bscanmeta_items(self) -> list[Type10004]:
        items = []
        for slice_structure in self._sorted_slices():
            folder = self._first_folder(slice_structure, TypesEnum.bscanmeta)
            if folder is not None:
                items.append(folder.data)
        return items

    def _bscanmeta_item(self, bscan_index: int) -> Type10004:
        items = self._bscanmeta_items()
        if bscan_index < 0 or bscan_index >= len(items):
            raise IndexError(
                f'bscan_index {bscan_index} is out of range for {len(items)} B-scans.'
            )
        return items[bscan_index]

    def _first_type39_raw(self) -> Optional[bytes]:
        for slice_structure in self._sorted_slices():
            folder = self._first_folder(slice_structure, TypesEnum.localizer_settings)
            if folder is not None:
                return folder.data.raw
        return None

    def _type39_uint8(self, offset: int) -> Optional[int]:
        raw = self._first_type39_raw()
        if raw is None or len(raw) <= offset:
            return None
        return raw[offset]

    def _type39_uint32(self, offset: int) -> Optional[int]:
        raw = self._first_type39_raw()
        if raw is None or len(raw) < offset + 4:
            return None
        return struct.unpack_from('<I', raw, offset)[0]

    def _type39_version(self, offset: int) -> Optional[str]:
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

    def _camera_model(self) -> Optional[str]:
        for text in self._type13_strings():
            if 'Spectralis' in text:
                return re.sub(r'\s*\+\s*', '+', text).replace('HRA+', 'HRA+')
        return None

    def _application(self) -> Optional[str]:
        examined_structure = self._series_text(TypesEnum.examined_structure, 0)
        for text in self._type13_strings():
            if text == examined_structure:
                return text
        return examined_structure

    def _series_text(self, folder_type: Union[TypesEnum, int],
                     index: int = 0) -> Optional[str]:
        folder = self._first_series_folder(folder_type)
        if folder is None:
            return None
        text = folder.data.text
        if index >= len(text):
            return None
        return text[index]

    def _study_text(self, folder_type: Union[TypesEnum, int],
                    index: int = 0) -> Optional[str]:
        folder = self._first_study_folder(folder_type)
        if folder is None:
            return None
        text = folder.data.text
        if index >= len(text):
            return None
        return text[index]

    def _series_exam_time_iso(self) -> Optional[str]:
        folder = self._first_series_folder(TypesEnum.examination_time)
        if folder is None:
            return None
        return folder.data.examination_time

    def _series_date_iso(self) -> Optional[str]:
        exam_time = self._series_exam_time_iso()
        if exam_time is None:
            return None
        return datetime.datetime.fromisoformat(exam_time).date().isoformat()

    def get_focus_candidate_raw(self) -> Optional[float]:
        items = self._bscanmeta_items()
        if not items:
            return None
        return items[0].focus_candidate_raw

    def _derive_scan_focus_diopters(self) -> Optional[float]:
        raw_value = self.get_focus_candidate_raw()
        if raw_value is None:
            return None
        return 1.079 * math.copysign(max(abs(raw_value) - 3.505, 0.0), raw_value)

    def get_x_scale_derivation(self) -> dict[str, Optional[float]]:
        items = self._bscanmeta_items()
        if not items:
            return {
                'angular_width_deg': None,
                'img_size_width': None,
                'constant_um_per_deg': MEDIUM_EYE_LENGTH_UM_PER_DEG,
                'scaling_x_um_per_pixel': None,
            }

        bscan_meta = items[0]
        angular_width_deg = abs(bscan_meta.end_x - bscan_meta.start_x)
        scaling_x = (angular_width_deg / bscan_meta.imgSizeWidth *
                     MEDIUM_EYE_LENGTH_UM_PER_DEG)
        return {
            'angular_width_deg': angular_width_deg,
            'img_size_width': bscan_meta.imgSizeWidth,
            'constant_um_per_deg': MEDIUM_EYE_LENGTH_UM_PER_DEG,
            'scaling_x_um_per_pixel': scaling_x,
        }

    def _oct_quality_db(self, bscan_index: Optional[int]) -> Optional[int]:
        if bscan_index is None:
            return None
        return round(self._bscanmeta_item(bscan_index).quality_ui)

    def _oct_art_mode(self, bscan_index: Optional[int]) -> Optional[int]:
        if bscan_index is None:
            return None
        return self._bscanmeta_item(bscan_index).art_mode

    def _oct_acquisition_time(self,
                              bscan_index: Optional[int]) -> Optional[str]:
        if bscan_index is None:
            return None
        return self._bscanmeta_item(bscan_index).acquisitionTime

    def _bscan_vertical_span_deg(self) -> Optional[float]:
        items = self._bscanmeta_items()
        if not items:
            return None
        centers = [item.center_y for item in items]
        return max(centers) - min(centers)

    def _build_heyex_metadata(
        self,
        bscan_index: Optional[int] = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        patient_folder = self._first_patient_folder(TypesEnum.patient)
        patient_data = patient_folder.data if patient_folder is not None else None
        exam_time_iso = self._series_exam_time_iso()
        exam_time = _format_time(exam_time_iso)
        focus_diopters = self._derive_scan_focus_diopters()
        focus_text = _format_scan_focus(focus_diopters)
        x_scale_info = self.get_x_scale_derivation()
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
            oct_scan_angle = abs(bscan_meta.end_x - bscan_meta.start_x)
            size_x = bscan_meta.size_x
            size_z = bscan_meta.size_y
            scaling_z = bscan_meta.scale_y * 1000

        pattern_height_mm = (None if pattern_height_deg is None else
                             pattern_height_deg * MEDIUM_EYE_LENGTH_UM_PER_DEG /
                             1000)
        distance_between_bscans_um = None
        if pattern_height_deg is not None and self.n_bscans > 1:
            distance_between_bscans_um = (
                pattern_height_deg * MEDIUM_EYE_LENGTH_UM_PER_DEG /
                (self.n_bscans - 1))

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
                'Scan Angle': '30°' if localizer_size_x is not None else None,
                'Size X': _format_px_mm(localizer_size_x, scaling_x)
                if localizer_size_x is not None else None,
                'Size Y': _format_px_mm(localizer_size_y, scaling_x)
                if localizer_size_y is not None else None,
                'Scaling': None if scaling_x is None else f'{scaling_x:.2f} µm/pixel',
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
                'Eye Length': None,
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
                'Scan Focus': 'provisional derived from Type10004@bscan0+offset140:float32',
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
                'Scaling': 'provisional derived from Type10004.start_x/end_x/imgSizeWidth',
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
                'Scaling X': 'provisional derived from Type10004.start_x/end_x/imgSizeWidth',
                'Scaling Z': 'Type10004.scale_y',
                'ART Mode': None if bscan_index is None else
                f'Type10004@bscan{bscan_index}+offset120:uint32',
                'A-Scan Rate': None,
                'Eye Length': None,
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
                'Distance between B-Scans': 'derived from Type10004 centers and provisional Medium eye-length constant',
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
        bscan_index: Optional[int] = None,
    ) -> dict[str, dict[str, Any]]:
        metadata, _ = self._build_heyex_metadata(bscan_index=bscan_index)
        return metadata

    def get_heyex_metadata_sources(
        self,
        bscan_index: Optional[int] = None,
    ) -> dict[str, dict[str, Any]]:
        _, sources = self._build_heyex_metadata(bscan_index=bscan_index)
        return sources

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

    def get_volume(self) -> EyeVolume:
        """Return EyeVolume object for the series."""
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

        data = self.get_bscans()

        volume_meta = self.get_meta()
        localizer = self.get_localizer()
        volume = EyeVolume(
            data=data,
            meta=volume_meta,
            localizer=localizer,
            transformation=_compute_localizer_oct_transform(
                volume_meta, localizer.meta, data.shape),
        )

        layer_height_maps = self.get_layers()
        for name, i in SEG_MAPPING.items():
            if i in layer_height_maps:
                volume.add_layer_annotation(layer_height_maps[i], name=name)

        return volume

    @property
    def n_bscans(self) -> int:
        """Return the number of B-scans in the series."""
        return len(self.substructure)

    def get_bscans(self) -> np.ndarray:
        volume_meta = self.get_meta()
        size_x = volume_meta['bscan_meta'][0]['size_x']
        size_y = volume_meta['bscan_meta'][0]['size_y']

        data = np.zeros((self.n_bscans, size_y, size_x), dtype=np.float32)
        for ind, sl in self.slices.items():
            bscan = sl.get_bscan()
            i = ind // 2 if len(
                self.get_bscan_meta()
            ) != 1 else 0  # Slice id for single B-scan Volumes is 2 and not 0 in the e2e file.

            data[i] = bscan

        return data

    def get_layers(self) -> dict[int, np.ndarray]:
        """Return layer height maps for the series as dict of numpy arrays where
        the key is the layer id."""
        slice_layers = {}
        layer_ids = set()

        for ind, sl in self.slices.items():
            layers = sl.get_layers()
            [layer_ids.add(k) for k in layers.keys()]
            slice_layers[ind // 2] = layers

        layers = {}
        size_x = self.get_bscan_meta()[0]['size_x']
        for i in layer_ids:
            layer = np.full((self.n_bscans, size_x), np.nan)
            if self.n_bscans == 1:
                layer[0, :] = slice_layers[1][i]
                layers[i] = layer

            else:
                for sl in range(self.n_bscans):
                    try:
                        layer[sl, :] = slice_layers[sl][i]
                    except KeyError:
                        pass

            layer[layer >= 3.0e+38] = np.nan
            layers[i] = layer

        return layers

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
            self._bscan_meta = sorted(
                [sl.get_meta() for sl in self.slices.values()],
                key=lambda x: x['aktImage'])
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
        self.folders: dict[Union[int, str], list[E2EFolder]] = {}
        self.patient: Optional[E2EPatientStructure] = None

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
        self.folders: dict[Union[int, str], list[E2EFolder]] = {}

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
        self.folders: dict[Union[int, str], list[E2EFolder]] = {}

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

    def __init__(self, path: Union[str, Path]):
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

    def find_int(self,
                 value: int,
                 excluded_folders: list[Union[int,
                                              str]] = ['images', 'layers'],
                 slice_id: Optional[int] = None,
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
                   excluded_folders: list[Union[int,
                                                str]] = ['images', 'layers'],
                   slice_id: Optional[int] = None,
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
                    value: Union[int, float],
                    excluded_folders: list[Union[int,
                                                 str]] = ['images', 'layers'],
                    slice_id: Optional[int] = None,
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
