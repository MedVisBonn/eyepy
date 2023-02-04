from collections import defaultdict
import logging
from typing import List, Tuple, Union

import construct as cs
import numpy as np

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.eyevolume import EyeVolume
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import get_bscan_spacing

from .e2e_format import e2e_format
from .vol_reader import SEG_MAPPING

logger = logging.getLogger(__name__)


def split_e2e(parsed_file) -> Tuple[List]:
    """Split an e2e file into a list of e2e files.

    We split e2e files with possibly multiple volumes into a list of e2e files of single volumes.
    Therefore we prepend and append patient and study information as in the original file but only keep series data of a single volume.

    Steps are:
    sorting all folders
    put folders in new chunks
        set chunk header according to content
    assemble new e2e file


    Args:
        parsed_file (e2e_format): The parsed e2e file.

    Returns:
        List: A list of parsed e2e files.
    """
    e2e_volumes = []
    e2e_enface = []

    general_data = []
    patient_data = []
    study_data = []
    series_dict = defaultdict(list)

    for chunk in parsed_file.chunks:
        for folder in chunk.folders:
            patient = folder.header.patient_id
            study = folder.header.study_id
            series = folder.header.series_id
            sli = folder.header.slice_id
            t = folder.header.type

            if patient != -1 and study != -1:
                # Series specific data or slice data
                if series != -1:
                    series_dict[series].append(folder)
                else:
                    study_data.append(folder)
            elif patient != -1 and study == -1:
                patient_data.append(folder)
            elif patient == -1:
                # General data
                if t == 9011:
                    general_data.append(folder)
                # Ignore chunk filler folders
                elif t == "empty_folder":
                    filler = folder
                else:
                    print(t)

                    print("Unknown folder", folder.header)
            else:
                print("Unknown folder", folder.header)

    for key, series_data in series_dict.items():
        folders = study_data + series_data + patient_data + general_data
        if "bscanmeta" in [f.header.type for f in series_data]:
            e2e_volumes.append(folders)
        else:
            e2e_enface.append(folders)

    return e2e_volumes, e2e_enface


class HeE2eReader:

    def __init__(self, path, single=True):
        self.path = path
        with open(self.path, "rb") as e2e_file:
            self.parsed_file = e2e_format.parse_stream(e2e_file)
            self._single_volumes, self._single_series = split_e2e(
                self.parsed_file
            )  # Split into single series (volumes and series without bscanmeta-probably not volumes)
            if len(self._single_volumes) > 1 and single:
                logger.warning(
                    "File contains more than one series. If you want to read all series, set single=False."
                )
            self.single = single

            self._meta = None
            self._bscan_meta = None

            self._current_volume_index = 0
            self._sort_folders(self.current_volume_index)
            #self._current_fundus_index = 0
            #self._sort_fundus_folders(self.current_fundus_index)

    @property
    def current_volume_index(self):
        return self._current_volume_index

    @current_volume_index.setter
    def current_volume_index(self, value):
        if 0 <= value < len(self._single_volumes):
            self._current_volume_index = value
            self._sort_folders(self._current_volume_index)
            self._meta = None
            self._bscan_meta = None
        else:
            logger.warning(
                f"Can not set current volume to {value}. There are only {len(self._single_volumes)} volumes in the file."
            )

    def _sort_folders(self, index):
        self._folders = defaultdict(list)
        for folder in self._single_volumes[index]:
            t = folder.header.type
            s = "enface" if folder.header.ind == 0 else "bscan"
            self._folders[t, s].append(folder)

    """
    @property
    def current_fundus_index(self):
        return self._current_fundus_index

    @current_fundus_index.setter
    def current_fundus_index(self, value):
        if 0 <= value < len(self._single_fundus):
            self._current_fundus_index = value
            self._sort_fundus_folders(self._current_volume_index)
        else:
            logger.warning(
                f"Can not set current fundus to {value}. There are only {len(self._single_volumes)} fundus images in the file."
            )

    def _sort_fundus_folders(self, index):
        self._fundus_folders = defaultdict(list)
        for folder in self._single_fundus[index]:
            t = folder.header.type
            self._fundus_folders[t].append(folder)
    """

    @property
    def volume(self) -> Union[EyeVolume, List[EyeVolume]]:
        if self.single:
            return self._get_volume()
        else:
            v = []
            for i in range(len(self._single_volumes)):
                self.current_volume_index = i
                v.append(self._get_volume())
            return v

    def _get_volume(self) -> EyeVolume:
        ## Check if scan is a volume scan
        volume_meta = self.meta

        size_x = volume_meta["bscan_meta"][0]["size_x"]
        size_y = volume_meta["bscan_meta"][0]["size_y"]
        scan_pattern = volume_meta["bscan_meta"][0]["scan_pattern"]
        n_bscans = volume_meta["bscan_meta"][0]["n_bscans"] if len(
            self.bscan_meta
        ) != 1 else 1  # n_bscans is 0 instead of 1 for single B-scan Volumes in the e2e file.

        # Check if scan is a volume scan
        if not scan_pattern in [1, 3, 4]:
            msg = f"Only volumes with ScanPattern 1, 3 or 4 are supported. The ScanPattern is {scan_pattern} which might lead to exceptions or unexpected behaviour."
            logger.warning(msg)

        data = np.zeros((n_bscans, size_y, size_x))
        folders = self._folders["image", "bscan"]
        for folder in folders:
            if folder.data_container.item.type == 35652097:
                s = folder.data_container.header.slice_id // 2 if len(
                    self.bscan_meta
                ) != 1 else 0  # Slice id for single B-scan Volumes is 2 and not 0 in the e2e file.
                data[s] = folder.data_container.item.data

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
            if i in layer_height_maps:
                volume.add_layer_annotation(layer_height_maps[i], name=name)

        return volume

    @property
    def layers(self):
        layer_segmentations = {}

        folders = self._folders["layer", "bscan"]
        idset = set()

        for folder in folders:
            item = folder.data_container.item
            layer_segmentations[item.id,
                                folder.header.slice_id / 2] = item.data
            idset.add(item.id)

        layers = {}
        n_bscans = self.bscan_meta[0]["n_bscans"]
        size_x = self.bscan_meta[0]["size_x"]
        for i in idset:
            layer = np.full((n_bscans, size_x), np.nan)
            for sl in range(n_bscans):
                layer[i, :] = layer_segmentations[i, sl]

            layers[i] = layer

        return layers

    @property
    def enface_modality(self):
        folders = self._folders["enface_modality", "enface"]
        if len(folders) > 1:
            logger.debug(
                "There is more than one enface modality stored. This is not expected."
            )
        text = folders[0].data_container.item.text[1]
        return "NIR" if text == "IR" else text

    @property
    def laterality(self):
        folders = self._folders["laterality", "enface"]
        if len(folders) > 1:
            logger.debug(
                "There is more than one laterality stored. This is not expected."
            )
        return str(folders[0].data_container.item.laterality)

    @property
    def localizer_meta(self) -> EyeEnfaceMeta:

        return EyeEnfaceMeta(
            scale_x=0.0114,  # Todo: Where is this in E2E?
            scale_y=0.0114,  # Todo: Where is this in E2E?
            scale_unit="mm",
            modality=self.enface_modality,
            laterality=self.laterality,
            field_size=None,
            scan_focus=None,
            visit_date=None,
            exam_time=None,
        )

    @property
    def localizer(self) -> EyeEnface:
        folders = self._folders["image", "enface"]
        if len(folders) > 1:
            logger.warning(
                "There is more than one enface localizer image stored. This is not expected."
            )
        localizer = folders[0].data_container.item.data
        return EyeEnface(localizer, self.localizer_meta)

    """
    @property
    def fundus(self) -> EyeEnface:
        if self.single:
            return self._get_fundus()
        else:
            f = []
            for i in range(len(self._single_enface)):
                self.current_volume_index = i
                f.append(self._get_fundus())
            return f

    def _get_fundus(self):
        folders = self._fundus_folders["fundus"]
        if len(folders) > 1:
            logger.warning(
                "There is more than one enface localizer image stored. This is not expected."
            )
        return EyeEnface(folders[0].data_container.item.data,
                         EyeEnfaceMeta(scale_unit="px", scale_x=1, scale_y=1))
    """

    @property
    def bscan_meta(self) -> List[EyeBscanMeta]:
        if self._bscan_meta is None:
            meta_objects = []
            folders = self._folders["bscanmeta", "bscan"]

            meta_objects = [
                EyeBscanMeta(  #quality=meta.quality,
                    start_pos=((meta.pop("start_x") + 14.86 * 0.29),
                               (meta.pop("start_y") + 15.02) * 0.29),
                    end_pos=((meta.pop("end_x") + 14.86 * 0.29),
                             (meta.pop("end_y") + 15.02) * 0.29),
                    pos_unit="mm",
                    **meta)
                for meta in [f.data_container.item for f in folders]
            ]
            #return meta_objects
            self._bscan_meta = sorted(meta_objects,
                                      key=lambda x: x["aktImage"])
        return self._bscan_meta

    @property
    def meta(self) -> EyeVolumeMeta:
        if self._meta is None:
            bscan_meta = self.bscan_meta
            self._meta = EyeVolumeMeta(
                scale_x=0.0114,  # Todo: Where is this in E2E?
                scale_y=self.bscan_meta[0]["scale_y"],
                scale_z=get_bscan_spacing(bscan_meta) if
                (bscan_meta[0]["scan_pattern"] not in [1, 2]) else 0.03,
                scale_unit="mm",
                laterality=self.laterality,
                visit_date=None,
                exam_time=None,
                bscan_meta=bscan_meta,
                intensity_transform="vol",
            )
        return self._meta
