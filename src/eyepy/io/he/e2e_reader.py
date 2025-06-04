from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
import dataclasses
from io import BufferedReader
import logging
from pathlib import Path
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
from eyepy.io.he.e2e_format import \
    containerheader_format  # data_container_structure, container_header_structure
from eyepy.io.he.e2e_format import datacontainer_format
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import find_float
from eyepy.io.utils import find_int
from eyepy.io.utils import get_bscan_spacing

from .e2e_format import ContainerHeader
from .e2e_format import DataContainer
from .e2e_format import e2e_format
from .e2e_format import Type10025
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
    reader: 'HeE2eReader'

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
        return EyeBscanMeta(  #quality=meta.quality,
            start_pos=((meta['start_x'] + 14.86 * 0.29),
                       (meta['start_y'] + 15.02) * 0.29),
            end_pos=((meta['end_x'] + 14.86 * 0.29),
                     (meta['end_y'] + 15.02) * 0.29),
            pos_unit='mm',
            **dataclasses.asdict(meta))

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

        data = np.zeros((self.n_bscans, size_y, size_x))
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
                    layer[sl, :] = slice_layers[sl][i]

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

    def localizer_meta(self) -> EyeEnfaceMeta:
        """Return EyeEnfaceMeta object for the localizer image."""
        if self._localizer_meta is None:
            self._localizer_meta = EyeEnfaceMeta(
                scale_x=1,  #0.0114,  # Todo: Where is this in E2E?
                scale_y=1,  #0.0114,  # Todo: Where is this in E2E?
                scale_unit='px',
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
            return EyeEnface(folders[0].data.data, self.localizer_meta())
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
                intensity_transform='vol',
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

    def __enter__(self) -> 'HeE2eReader':
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
