import logging
from pathlib import Path
from typing import List, Union

import imageio.v2 as imageio
import numpy as np
from oct_converter.readers import FDA

from eyepy import EyeBscanMeta
from eyepy import EyeVolume
from eyepy import EyeVolumeMeta
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.io.utils import _compute_localizer_oct_transform

from .he import HeE2eReader
from .he import HeVolReader
from .he import HeVolWriter
from .he import HeXmlReader

logger = logging.getLogger('eyepy.io')


def import_topcon_fda(path: Union[str, Path]) -> EyeVolume:
    """Read a Topcon fda file.

    This function is a wrapper around the FDA reader in OCT-Converter.

    Args:
        path: Path to the fda file

    Returns:
        Parsed data as EyeVolume object

    Notes
    -----
    B-scan position and scaling data is computed assuming that B-scans
    were acquired in a horizontal raster pattern.
    """
    reader = FDA(path, printing=False)

    try:
        oct_volume = reader.read_oct_volume()
        segmentation = oct_volume.contours
        metadata = oct_volume.metadata
    except:
        logger.warn('Regular B-scan read failed. Using alternative.')
        oct_volume = reader.read_oct_volume_2()
        segmentation = reader.read_segmentation()
        metadata = reader.read_all_metadata()

    localizer_image = reader.read_fundus_image().image
    bscan = oct_volume.volume

    # retrieve image dimensions and scaling
    n_bscan = len(bscan)
    n_axial = bscan[0].shape[0]
    n_ascan = bscan[0].shape[1]

    size_x = metadata['param_scan_04']['x_dimension_mm']
    size_z = metadata['param_scan_04']['y_dimension_mm']

    scale_x = size_x / (n_ascan - 1)
    scale_y = metadata['param_scan_04']['z_resolution_um'] / 1000
    scale_z = size_z / (n_bscan - 1)

    # compute B-scan mm coordinates from fundus top-left corner
    box = metadata['regist_info']['bounding_box_in_fundus_pixels']

    scale_x_fun = size_x / (box[2] - box[0] - 1)
    scale_z_fun = size_z / (box[3] - box[1] - 1)

    x_0 = box[0] * scale_x_fun  # top-left x
    z_0 = box[1] * scale_z_fun  # top-left y
    x_1 = box[2] * scale_x_fun  # top-right x

    # build metadata objects
    bscan_meta = []
    for i in range(n_bscan):
        z = z_0 + i * scale_z
        bscan_meta.append(
            EyeBscanMeta(start_pos=(x_0, z), end_pos=(x_1, z), pos_unit='mm'))

    volume_meta = EyeVolumeMeta(scale_x=scale_x,
                                scale_y=scale_y,
                                scale_z=scale_z,
                                scale_unit='mm',
                                bscan_meta=bscan_meta)

    localizer_meta = EyeEnfaceMeta(scale_x=scale_x_fun,
                                   scale_y=scale_z_fun,
                                   scale_unit='mm',
                                   modality='CFP',
                                   laterality='unknown')

    # build image ojects
    localizer = EyeEnface(localizer_image, localizer_meta)
    dims = (n_bscan, n_axial, n_ascan)
    transformation = _compute_localizer_oct_transform(volume_meta,
                                                      localizer.meta, dims)

    ev = EyeVolume(data=np.stack(bscan),
                   meta=volume_meta,
                   localizer=localizer,
                   transformation=transformation)

    if segmentation:
        for name, i in segmentation.items():
            ev.add_layer_annotation(segmentation[name], name=name)

    return ev


def import_heyex_e2e(path: Union[str, Path]) -> EyeVolume:
    """Read a Heyex E2E file.

    This function is a thin wrapper around the HeE2eReader class and
    returns the first of potentially multiple OCT volumes. If you want to
    read all volumes, or need more control, you can use the
    [HeE2eReader][eyepy.io.he.e2e_reader.HeE2eReader] class directly.

    Args:
        path: Path to the E2E file

    Returns:
        Parsed data as EyeVolume object
    """
    reader = HeE2eReader(path)
    if len(reader.series) < 1:
        logger.info(
            f'There are {len(reader.series)} Series stored in the E2E file. If you want to read all of them, use the HeE2eReader class directly.'
        )
    with reader as open_reader:
        ev = open_reader.volume
    return ev


def import_heyex_xml(path: Union[str, Path]) -> EyeVolume:
    """Read a Heyex XML file.

    This function is a thin wrapper around the HeXmlReader class
    which you can use directly if you need more control.

    Args:
        path: Path to the XML file or the folder containing the XML file

    Returns:
        Parsed data as EyeVolume object
    """
    return HeXmlReader(path).volume


def import_heyex_vol(path: Union[str, Path]) -> EyeVolume:
    """Read a Heyex VOL file.

    This function is a thin wrapper around the HeVolReader class
    which you can use directly if you need more control.

    Args:
        path: Path to the VOL file

    Returns:
        Parsed data as EyeVolume object
    """
    return HeVolReader(path).volume


def import_heyex_angio_vol(path: Union[str, Path]) -> EyeVolume:
    """Read a Heyex Angio VOL file.

    This function is a thin wrapper around the HeVolReader class
    which you can use directly if you need more control.

    Args:
        path: Path to the Angio VOL file

    Returns:
        Parsed data as EyeVolume object
    """
    return HeVolReader(path, type='octa').volume


def import_bscan_folder(path: Union[str, Path]) -> EyeVolume:
    """Read B-Scans from a folder.

    This function can be used to read B-scans from a folder in case that
    there is no additional metadata available.

    Args:
        path: Path to the folder containing the B-Scans

    Returns:
        Parsed data as EyeVolume object
    """
    path = Path(path)
    img_paths = sorted(list(path.iterdir()))
    img_paths = [
        p for p in img_paths if p.is_file()
        and p.suffix.lower() in ['.jpg', '.jpeg', '.tiff', '.tif', '.png']
    ]

    images = []
    for p in img_paths:
        image = imageio.imread(p)
        if len(image.shape) == 3:
            image = image[..., 0]
        images.append(image)

    volume = np.stack(images, axis=0)
    bscan_meta = [
        EyeBscanMeta(start_pos=(0, i),
                     end_pos=(volume.shape[2] - 1, i),
                     pos_unit='pixel')
        for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeVolumeMeta(scale_x=1,
                         scale_y=1,
                         scale_z=1,
                         scale_unit='pixel',
                         bscan_meta=bscan_meta)

    return EyeVolume(data=volume, meta=meta)


def import_duke_mat(path: Union[str, Path]) -> EyeVolume:
    """Import an OCT volume from the Duke dataset.

    The dataset is available at https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm
    OCT volumes are stored as .mat files which are parsed by this function and returned as
    EyeVolume object.

    Args:
        path: Path to the .mat file

    Returns:
        Parsed data as EyeVolume object
    """
    import scipy.io as sio

    loaded = sio.loadmat(path)
    volume = np.moveaxis(loaded['images'], -1, 0)
    layer_maps = np.moveaxis(loaded['layerMaps'], -1, 0)

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, 0.067 * i),
            end_pos=(0.0067 * (volume.shape[2] - 1), 0.067 * i),
            pos_unit='mm',
        ) for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeVolumeMeta(
        scale_x=0.0067,
        scale_y=0.0045,  # https://retinatoday.com/articles/2008-may/0508_10-php
        scale_z=0.067,
        scale_unit='mm',
        bscan_meta=bscan_meta,
        age=int(loaded['Age'][0][0]),
    )

    volume = EyeVolume(data=volume, meta=meta)
    names = {0: 'ILM', 1: 'IBRPE', 2: 'BM'}
    for i, height_map in enumerate(layer_maps):
        volume.add_layer_annotation(
            height_map,
            name=names[i],
        )

    return volume


def import_dukechiu2_mat(path: Union[str, Path]) -> EyeVolume:
    """Import an OCT volume from the Duke dataset (Chiu_BOE_2014).

    The dataset is available at https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm
    OCT volumes are stored as .mat files which are parsed by this function and returned as
    EyeVolume object.

    Args:
        path: Path to the .mat file

    Returns:
        Parsed data as EyeVolume object
    """
    import scipy.io as sio

    loaded = sio.loadmat(path)
    volume = np.moveaxis(loaded['images'], -1, 0)

    layer_versions = [
        'manualLayers1', 'manualLayers2', 'automaticLayersDME',
        'automaticLayersNormal'
    ]
    fluid_versions = ['manualFluid1', 'manualFluid2', 'automaticFluidDME']
    all_layer_maps = {
        x: np.moveaxis(loaded[x], -1, 1).astype(np.float32)
        for x in layer_versions
    }
    # Manual annotations are instances of fluid regions, we convert to binary here
    all_pixel_maps = {
        x: (np.moveaxis(loaded[x], -1, 0) > 0).astype(bool)
        for x in fluid_versions
    }

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, 0.123 * i),
            end_pos=(0.01133 * (volume.shape[2] - 1), 0.123 * i),
            pos_unit='mm',
        ) for i in range(volume.shape[0] - 1, -1, -1)
    ]
    meta = EyeVolumeMeta(
        scale_x=
        0.01133,  # This value is the average of min and max values given in the paper.
        scale_y=0.00387,
        scale_z=
        0.123,  # This value is the average of min and max values given in the paper.
        scale_unit='mm',
        bscan_meta=bscan_meta,
    )

    volume = EyeVolume(data=volume, meta=meta)
    names = {
        7: 'BM',
        6: 'OS/RPE',
        5: 'ISM/ISE',
        4: 'OPL/ONL',
        3: 'INL/OPL',
        2: 'IPL/INL',
        1: 'NFL/GCL',
        0: 'ILM'
    }

    for layer_version in layer_versions:
        layer_maps = all_layer_maps[layer_version]
        for i, height_map in enumerate(layer_maps):
            volume.add_layer_annotation(
                height_map,
                name=f'{layer_version}_{names[i]}',
            )

    for fluid_version in fluid_versions:
        pixel_map = all_pixel_maps[fluid_version]
        for i, height_map in enumerate(pixel_map):
            volume.add_pixel_annotation(
                pixel_map,
                name=fluid_version,
            )

    return volume


def import_retouch(path: Union[str, Path]) -> EyeVolume:
    """Import an OCT volume from the Retouch dataset.

    The dataset is available upon request at https://retouch.grand-challenge.org/
    Reading the data requires the ITK library to be installed. You can install it with pip:

    `pip install itk`

    Args:
        path: Path to the folder containing the OCT volume

    Returns:
        Parsed data as EyeVolume object
    """
    import itk

    path = Path(path)
    data = itk.imread(str(path / 'oct.mhd'))

    bscan_meta = [
        EyeBscanMeta(
            start_pos=(0, data['spacing'][0] * i),
            end_pos=(data['spacing'][2] * (data.shape[2] - 1),
                     data['spacing'][0] * i),
            pos_unit='mm',
        ) for i in range(data.shape[0] - 1, -1, -1)
    ]

    meta = EyeVolumeMeta(
        scale_x=data['spacing'][2],
        scale_y=data['spacing'][1],
        scale_z=data['spacing'][0],
        scale_unit='mm',
        bscan_meta=bscan_meta,
    )
    # Todo: Add intensity transform instead. Topcon and Cirrus are stored as UCHAR while heidelberg is stored as USHORT
    data = (data[...].astype(float) / np.iinfo(data[...].dtype).max *
            255).astype(np.uint8)
    eye_volume = EyeVolume(data=data[...], meta=meta)

    if (path / 'reference.mhd').is_file():
        annotation = itk.imread(str(path / 'reference.mhd'))
        eye_volume.add_pixel_annotation(np.equal(annotation, 1),
                                        name='IRF',
                                        current_color='FF0000')
        eye_volume.add_pixel_annotation(np.equal(annotation, 2),
                                        name='SRF',
                                        current_color='0000FF')
        eye_volume.add_pixel_annotation(np.equal(annotation, 3),
                                        name='PED',
                                        current_color='FFFF00')

    return eye_volume
