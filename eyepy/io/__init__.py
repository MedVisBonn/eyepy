import logging
from pathlib import Path

import imageio
import numpy as np

from eyepy import EyeBscanMeta, EyeEnface, EyeVolume, EyeVolumeMeta
from eyepy.io.lazy import LazyVolume
from eyepy.io.utils import (
    _compute_localizer_oct_transform,
    _get_enface_meta,
    _get_volume_meta,
)

logger = logging.getLogger("eyepy.io")


def import_heyex_xml(path):
    from eyepy.io.heyex import HeyexXmlReader

    reader = HeyexXmlReader(path)

    l_volume = LazyVolume(
        bscans=reader.bscans,
        localizer=reader.localizer,
        meta=reader.oct_meta,
        data_path=reader.path,
    )

    ## Check if scan is a volume scan
    if not l_volume.ScanPattern in [3, 4]:
        msg = f"Only volumes with ScanPattern 3 or 4 are supported. The ScanPattern is {l_volume.ScanPattern} which might lead to exceptions or unexpected behaviour."
        logger.warning(msg)

    enface_meta = _get_enface_meta(l_volume)
    volume_meta = _get_volume_meta(l_volume)
    transformation = _compute_localizer_oct_transform(
        volume_meta, enface_meta, l_volume.shape
    )

    if len(l_volume.localizer.shape) == 3:
        localizer = l_volume.localizer[..., 0]
    else:
        localizer = l_volume.localizer

    enface = EyeEnface(data=localizer, meta=enface_meta)
    volume = EyeVolume(
        data=l_volume.volume,
        meta=volume_meta,
        localizer=enface,
        transformation=transformation,
    )

    layer_height_maps = l_volume.layers
    for key, val in layer_height_maps.items():
        volume.add_layer_annotation(val, name=key)

    return volume


def import_heyex_vol(path):
    from eyepy.io.heyex import HeyexVolReader

    reader = HeyexVolReader(path)
    l_volume = LazyVolume(
        bscans=reader.bscans,
        localizer=reader.localizer,
        meta=reader.oct_meta,
        data_path=Path(path).parent,
    )

    ## Check if scan is a volume scan
    if not l_volume.ScanPattern in [1, 3, 4]:
        msg = f"Only volumes with ScanPattern 1, 3 or 4 are supported. The ScanPattern is {l_volume.ScanPattern} which might lead to exceptions or unexpected behaviour."
        logger.warning(msg)

    enface_meta = _get_enface_meta(l_volume)
    volume_meta = _get_volume_meta(l_volume)
    volume_meta["intensity_transform"] = "vol"
    transformation = _compute_localizer_oct_transform(
        volume_meta, enface_meta, l_volume.shape
    )

    enface = EyeEnface(data=l_volume.localizer, meta=enface_meta)
    volume = EyeVolume(
        data=l_volume.volume_raw,
        meta=volume_meta,
        localizer=enface,
        transformation=transformation,
    )

    volume.set_intensity_transform("vol")

    layer_height_maps = l_volume.layers
    for key, val in layer_height_maps.items():
        volume.add_layer_annotation(val, name=key)

    return volume


def import_bscan_folder(path):
    path = Path(path)
    img_paths = sorted(list(path.iterdir()))
    img_paths = [
        p
        for p in img_paths
        if p.is_file()
        and p.suffix.lower() in [".jpg", ".jpeg", ".tiff", ".tif", ".png"]
    ]

    images = []
    for p in img_paths:
        image = imageio.imread(p)
        if len(image.shape) == 3:
            image = image[..., 0]
        images.append(image)

    volume = np.stack(images, axis=0)
    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, i), end_pos=(volume.shape[2] - 1, i), pos_unit="pixel"
        )
        for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeVolumeMeta(
        scale_x=1, scale_y=1, scale_z=1, scale_unit="pixel", bscan_meta=bscan_meta
    )

    return EyeVolume(data=volume, meta=meta)


def import_duke_mat(path):
    import scipy.io as sio

    loaded = sio.loadmat(path)
    volume = np.moveaxis(loaded["images"], -1, 0)
    layer_maps = np.moveaxis(loaded["layerMaps"], -1, 0)

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, 0.067 * i),
            end_pos=(0.0067 * (volume.shape[2] - 1), 0.067 * i),
            pos_unit="mm",
        )
        for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeVolumeMeta(
        scale_x=0.0067,
        scale_y=0.0045,  # https://retinatoday.com/articles/2008-may/0508_10-php
        scale_z=0.067,
        scale_unit="mm",
        bscan_meta=bscan_meta,
        age=loaded["Age"],
    )

    volume = EyeVolume(data=volume, meta=meta)
    names = {0: "ILM", 1: "IBRPE", 2: "BM"}
    for i, height_map in enumerate(layer_maps):
        volume.add_layer_annotation(
            np.flip(height_map, axis=0),
            name=names[i],
        )

    return volume


def import_retouch(path):
    import itk

    path = Path(path)
    data = itk.imread(str(path / "oct.mhd"))

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, data["spacing"][0] * i),
            end_pos=(data["spacing"][2] * (data.shape[2] - 1), data["spacing"][0] * i),
            pos_unit="mm",
        )
        for i in range(data.shape[0] - 1, -1, -1)
    ]

    meta = EyeVolumeMeta(
        scale_x=data["spacing"][2],
        scale_y=data["spacing"][1],
        scale_z=data["spacing"][0],
        scale_unit="mm",
        bscan_meta=bscan_meta,
    )
    # Todo: Add intensity transform instead. Topcon and Cirrus are stored as UCHAR while spectralis is stored as USHORT
    data = (data[...].astype(float) / np.iinfo(data[...].dtype).max * 255).astype(
        np.uint8
    )
    eye_volume = EyeVolume(data=data[...], meta=meta)

    if (path / "reference.mhd").is_file():
        annotation = itk.imread(str(path / "reference.mhd"))
        eye_volume.add_voxel_annotation(
            np.equal(annotation, 1), name="IRF", current_color="FF0000"
        )
        eye_volume.add_voxel_annotation(
            np.equal(annotation, 2), name="SRF", current_color="0000FF"
        )
        eye_volume.add_voxel_annotation(
            np.equal(annotation, 3), name="PED", current_color="FFFF00"
        )

    return eye_volume
