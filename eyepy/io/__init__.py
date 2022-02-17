from pathlib import Path

from eyepy import (
    EyeMeta,
    EyeVolume,
    EyeEnface,
    EyeVolumeLayerAnnotation,
    EyeVolumeVoxelAnnotation,
)
from eyepy.io.utils import (
    _compute_localizer_oct_transform,
    _get_enface_meta,
    _get_volume_meta,
)
from eyepy.io.lazy import LazyVolume

import imageio
import numpy as np
import logging

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

    layer_height_maps = l_volume.layers
    layers = {
        key: EyeVolumeLayerAnnotation(val, key)
        for key, val in layer_height_maps.items()
    }

    enface_meta = _get_enface_meta(l_volume)
    volume_meta = _get_volume_meta(l_volume)
    transformation = _compute_localizer_oct_transform(volume_meta, enface_meta)

    if len(l_volume.localizer.shape) == 3:
        localizer = l_volume.localizer[..., 0]
    else:
        localizer = l_volume.localizer

    enface = EyeEnface(data=localizer, meta=enface_meta)
    volume = EyeVolume(
        data=l_volume.volume,
        meta=volume_meta,
        layers=layers,
        localizer=enface,
        transformation=transformation,
    )

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
    if not l_volume.ScanPattern in [3, 4]:
        msg = f"Only volumes with ScanPattern 3 or 4 are supported. The ScanPattern is {l_volume.ScanPattern} which might lead to exceptions or unexpected behaviour."
        logger.warning(msg)

    layer_height_maps = l_volume.layers
    layers = {
        key: EyeVolumeLayerAnnotation(val, key)
        for key, val in layer_height_maps.items()
    }
    enface_meta = _get_enface_meta(l_volume)
    volume_meta = _get_volume_meta(l_volume)
    transformation = _compute_localizer_oct_transform(volume_meta, enface_meta)

    enface = EyeEnface(data=l_volume.localizer, meta=enface_meta)
    volume = EyeVolume(
        data=l_volume.volume,
        meta=volume_meta,
        layers=layers,
        localizer=enface,
        transformation=transformation,
    )

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
        EyeMeta(
            start_pos=(0, i),
            end_pos=(volume.shape[2] - 1, i),
        )
        for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeMeta(
        size_x=volume.shape[2],
        size_y=volume.shape[1],
        size_z=volume.shape[0],
        scale_x=1,
        scale_y=1,
        scale_z=1,
        bscan_meta=bscan_meta,
    )

    return EyeVolume(data=volume, meta=meta)


def import_duke_mat(path):
    import scipy.io as sio

    loaded = sio.loadmat(path)
    volume = np.moveaxis(loaded["images"], -1, 0)
    layer_maps = np.moveaxis(loaded["layerMaps"], -1, 0)
    names = {0: "ILM", 1: "IBRPE", 2: "BM"}
    layers = {
        names[i]: EyeVolumeLayerAnnotation(np.flip(height_map, axis=0), name=names[i])
        for i, height_map in enumerate(layer_maps)
    }

    bscan_meta = [
        EyeMeta(
            start_pos=(0, 0.067 * i),
            end_pos=(0.0067 * (volume.shape[2] - 1), 0.067 * i),
        )
        for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeMeta(
        size_x=volume.shape[2],
        size_y=volume.shape[1],
        size_z=volume.shape[0],
        scale_x=0.0067,
        scale_y=0.0045,  # https://retinatoday.com/articles/2008-may/0508_10-php
        scale_z=0.067,
        bscan_meta=bscan_meta,
        age=loaded["Age"],
    )

    return EyeVolume(data=volume, meta=meta, layers=layers)


def import_retouch(path):
    import itk

    path = Path(path)
    data = itk.imread(str(path / "oct.mhd"))
    annotation = itk.imread(str(path / "reference.mhd"))

    bscan_meta = [
        EyeMeta(
            start_pos=(0, data["spacing"][0] * i),
            end_pos=(data["spacing"][2] * (data.shape[2] - 1), data["spacing"][0] * i),
        )
        for i in range(data.shape[0] - 1, -1, -1)
    ]

    meta = EyeMeta(
        size_x=data.shape[2],
        size_y=data.shape[1],
        size_z=data.shape[0],
        scale_x=data["spacing"][2],
        scale_y=data["spacing"][1],
        scale_z=data["spacing"][0],
        bscan_meta=bscan_meta,
    )

    eye_volume = EyeVolume(data=data[...], meta=meta)
    eye_volume.set_volume_map("IRF", np.equal(annotation, 1))
    eye_volume.set_volume_map("SRF", np.equal(annotation, 2))
    eye_volume.set_volume_map("PED", np.equal(annotation, 3))

    return eye_volume
