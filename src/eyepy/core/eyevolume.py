from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import io
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Optional, overload, SupportsIndex, Union
import warnings
import zipfile

from matplotlib import patches
import matplotlib.pyplot as plt
from numpy import typing as npt
import numpy as np
from skimage import transform
from skimage.transform._geometric import _GeometricTransform

from eyepy import config
from eyepy.core.annotations import EyeVolumeLayerAnnotation
from eyepy.core.annotations import EyeVolumePixelAnnotation
from eyepy.core.annotations import EyeVolumeSlabAnnotation
from eyepy.core.eyebscan import EyeBscan
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.utils import intensity_transforms
from eyepy.core.utils import par_algorithms

logger = logging.getLogger('eyepy.core.eyevolume')


class EyeVolume:
    """"""

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        meta: Optional[EyeVolumeMeta] = None,
        localizer: Optional[EyeEnface] = None,
        transformation: Optional[_GeometricTransform] = None,
    ) -> None:
        """

        Args:
            data: A 3D numpy array containing the OCT data in shape (n_bscans, bscan_height, bscan_width)
            meta: Optional [EyeVolumeMeta][eyepy.core.eyemeta.EyeVolumeMeta] object.
            localizer:
            transformation:
        """
        self._raw_data = data
        self._data = None
        self._data_par = None

        self._bscans = {}

        if meta is None:
            self.meta = self._default_meta(self._raw_data)
        else:
            self.meta = meta
        if 'intensity_transform' not in self.meta:
            self.meta['intensity_transform'] = 'default'
        if 'par_algorithm' not in self.meta:
            self.meta['par_algorithm'] = 'default'

        self.set_intensity_transform(self.meta['intensity_transform'])
        self.set_par_algorithm(self.meta['par_algorithm'])

        self._slabs = []
        self._layers = []
        self._volume_maps = []
        self._ascan_maps = []

        if transformation is None:
            self.localizer_transform = self._estimate_transform()
        else:
            self.localizer_transform = transformation

        if localizer is None:
            self.localizer = self._default_localizer(self.data)
        else:
            self.localizer = localizer

    def save(self, path: Union[str, Path], compress: bool = False) -> None:
        """Save the EyeVolume to a zip file.

        Args:
            path: Path where the file will be saved
            compress: Whether to compress the file. Compression reduces file size by ~38% but takes ~30x longer.
                     Default is False for fast saves. Set to True to reduce file size. (default: False)

        Returns:
            None
        """
        path = Path(path)

        compression_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        compress_level = 6 if compress else None

        with zipfile.ZipFile(path, 'w', compression_type, compresslevel=compress_level) as zipf:
            # Save OCT volume as npy and meta as json
            volume_bytes = io.BytesIO()
            np.save(volume_bytes, self._raw_data)
            zipf.writestr('raw_volume.npy', volume_bytes.getvalue())

            meta_dict = self.meta.as_dict()
            if meta_dict['intensity_transform'] == 'custom':
                warnings.warn(
                    'Custom intensity transforms can not be saved.')
                meta_dict['intensity_transform'] = 'default'
            zipf.writestr('meta.json', json.dumps(meta_dict))

            # Save Volume Annotations
            if len(self._volume_maps) > 0:
                voxel_data_bytes = io.BytesIO()
                np.save(
                    voxel_data_bytes,
                    np.stack([v.data for v in self._volume_maps]),
                )
                zipf.writestr('annotations/voxels/voxel_maps.npy', voxel_data_bytes.getvalue())
                zipf.writestr(
                    'annotations/voxels/meta.json',
                    json.dumps([v.meta for v in self._volume_maps])
                )

            # Save layer annotations
            if len(self._layers) > 0:
                layer_data_bytes = io.BytesIO()
                np.save(
                    layer_data_bytes,
                    np.stack([l.data for l in self._layers]),
                )
                zipf.writestr('annotations/layers/layer_heights.npy', layer_data_bytes.getvalue())
                zipf.writestr(
                    'annotations/layers/meta.json',
                    json.dumps([l.meta for l in self._layers])
                )

            # Save slab annotations
            if len(self._slabs) > 0:
                zipf.writestr(
                    'annotations/slabs/meta.json',
                    json.dumps([s.meta for s in self._slabs])
                )

            # Save Localizer
            localizer_bytes = io.BytesIO()
            np.save(localizer_bytes, self.localizer.data)
            zipf.writestr('localizer/localizer.npy', localizer_bytes.getvalue())
            zipf.writestr('localizer/meta.json', json.dumps(self.localizer.meta.as_dict()))

            # Save localizer transform
            if self.localizer_transform is not None:
                transform_bytes = io.BytesIO()
                np.save(transform_bytes, self.localizer_transform.params)
                zipf.writestr('localizer/transform_params.npy', transform_bytes.getvalue())

            # Save Localizer Annotations
            if len(self.localizer._area_maps) > 0:
                pixels_data_bytes = io.BytesIO()
                np.save(
                    pixels_data_bytes,
                    np.stack([p.data for p in self.localizer._area_maps]),
                )
                zipf.writestr('localizer/annotations/pixel/pixel_maps.npy', pixels_data_bytes.getvalue())
                zipf.writestr(
                    'localizer/annotations/pixel/meta.json',
                    json.dumps([p.meta for p in self.localizer._area_maps])
                )

            # Save Optic Disc annotation
            if self.localizer.optic_disc is not None:
                polygon_bytes = io.BytesIO()
                np.save(polygon_bytes, self.localizer.optic_disc.polygon)
                zipf.writestr('localizer/annotations/optic_disc/polygon.npy', polygon_bytes.getvalue())

                if self.localizer.optic_disc.shape is not None:
                    zipf.writestr(
                        'localizer/annotations/optic_disc/shape.json',
                        json.dumps(self.localizer.optic_disc.shape)
                    )

            # Save Fovea annotation
            if self.localizer.fovea is not None:
                fovea_polygon_bytes = io.BytesIO()
                np.save(fovea_polygon_bytes, self.localizer.fovea.polygon)
                zipf.writestr('localizer/annotations/fovea/polygon.npy', fovea_polygon_bytes.getvalue())

                if self.localizer.fovea.shape is not None:
                    zipf.writestr(
                        'localizer/annotations/fovea/shape.json',
                        json.dumps(self.localizer.fovea.shape)
                    )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EyeVolume':
        """Load an EyeVolume from a zip file.

        Args:
            path: Path to the zip file to load

        Returns:
            EyeVolume: The loaded EyeVolume object
        """
        path = Path(path)

        with zipfile.ZipFile(path, 'r') as zipf:
            # Load raw volume and meta
            with zipf.open('raw_volume.npy') as f:
                data = np.load(io.BytesIO(f.read()))
            with zipf.open('meta.json') as f:
                volume_meta = EyeVolumeMeta.from_dict(json.load(f))

            # Load Volume Annotations
            try:
                with zipf.open('annotations/voxels/voxel_maps.npy') as f:
                    voxel_annotations = np.load(io.BytesIO(f.read()))
                with zipf.open('annotations/voxels/meta.json') as f:
                    voxels_meta = json.load(f)
            except KeyError:
                voxel_annotations = []
                voxels_meta = []

            # Load layers
            try:
                with zipf.open('annotations/layers/layer_heights.npy') as f:
                    layer_annotations = np.load(io.BytesIO(f.read()))
                with zipf.open('annotations/layers/meta.json') as f:
                    layers_meta = json.load(f)

                # Clean knots
                for i, layer_meta in enumerate(layers_meta):
                    if 'knots' in layer_meta:
                        knots = layer_meta['knots']
                        knots = {int(i): knots[i] for i in knots}
                        layer_meta['knots'] = knots
            except KeyError:
                layer_annotations = []
                layers_meta = []

            # Load slabs
            try:
                with zipf.open('annotations/slabs/meta.json') as f:
                    slabs_meta = json.load(f)
            except KeyError:
                slabs_meta = []

            # Load Localizer and meta
            with zipf.open('localizer/localizer.npy') as f:
                localizer_data = np.load(io.BytesIO(f.read()))
            with zipf.open('localizer/meta.json') as f:
                localizer_meta = EyeEnfaceMeta.from_dict(json.load(f))

            # Load Optic Disc annotation if it exists
            optic_disc = None
            try:
                from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation
                with zipf.open('localizer/annotations/optic_disc/polygon.npy') as f:
                    polygon = np.load(io.BytesIO(f.read()))
                shape = None
                try:
                    with zipf.open('localizer/annotations/optic_disc/shape.json') as f:
                        shape = tuple(json.load(f))
                except KeyError:
                    pass
                optic_disc = EyeEnfaceOpticDiscAnnotation(polygon=polygon, shape=shape)
            except KeyError:
                pass

            # Load Fovea annotation if it exists
            fovea = None
            try:
                from eyepy.core.annotations import EyeEnfaceFoveaAnnotation
                with zipf.open('localizer/annotations/fovea/polygon.npy') as f:
                    polygon = np.load(io.BytesIO(f.read()))
                shape = None
                try:
                    with zipf.open('localizer/annotations/fovea/shape.json') as f:
                        shape = tuple(json.load(f))
                except KeyError:
                    pass
                fovea = EyeEnfaceFoveaAnnotation(polygon=polygon, shape=shape)
            except KeyError:
                pass

            localizer = EyeEnface(data=localizer_data, meta=localizer_meta,
                                 optic_disc=optic_disc, fovea=fovea)

            # Load Localizer Annotations
            try:
                with zipf.open('localizer/annotations/pixel/pixel_maps.npy') as f:
                    pixel_annotations = np.load(io.BytesIO(f.read()))
                with zipf.open('localizer/annotations/pixel/meta.json') as f:
                    pixels_meta = json.load(f)

                for i, pixel_meta in enumerate(pixels_meta):
                    localizer.add_area_annotation(pixel_annotations[i],
                                                  pixel_meta)
            except KeyError:
                pass

            # Load localizer transform if it exists, otherwise compute it
            try:
                with zipf.open('localizer/transform_params.npy') as f:
                    transform_params = np.load(io.BytesIO(f.read()))
                transformation = transform.AffineTransform(matrix=transform_params)
            except KeyError:
                # Backward compatibility: compute transform if not saved
                from eyepy.io.utils import _compute_localizer_oct_transform
                transformation = _compute_localizer_oct_transform(
                    volume_meta, localizer_meta, data.shape)

            ev = cls(
                data=data,
                meta=volume_meta,
                localizer=localizer,
                transformation=transformation,
            )
            for meta, annotation in zip(layers_meta, layer_annotations):
                ev.add_layer_annotation(annotation, meta)

            for meta in slabs_meta:
                ev.add_slab_annotation(meta)

            for meta, annotation in zip(voxels_meta, voxel_annotations):
                ev.add_pixel_annotation(annotation, meta)

        return ev

    def _default_meta(self, volume: npt.NDArray[np.float64]) -> EyeVolumeMeta:
        bscan_meta = [
            EyeBscanMeta(start_pos=(0, i),
                         end_pos=((volume.shape[2] - 1), i),
                         pos_unit='pixel')
            for i in range(volume.shape[0] - 1, -1, -1)
        ]
        meta = EyeVolumeMeta(
            scale_x=1.0,
            scale_y=1.0,
            scale_z=1.0,
            scale_unit='pixel',
            intensity_transform='default',
            bscan_meta=bscan_meta,
        )
        return meta

    def _default_localizer(self, data: npt.NDArray[np.float64]) -> EyeEnface:
        projection = np.flip(np.nanmean(data, axis=1), axis=0)
        image = transform.warp(
            projection,
            self.localizer_transform.inverse,
            output_shape=(self.size_x, self.size_x),
            order=1,
        )
        localizer = EyeEnface(
            image,
            meta=EyeEnfaceMeta(
                scale_x=self.scale_x,
                scale_y=self.scale_x,
                scale_unit=self.scale_unit,
                field_size=0,
                scan_focus=0,
                laterality='NA',
            ),
        )
        return localizer

    def _estimate_transform(self) -> transform.AffineTransform:
        # Compute a transform to map a 2D projection of the volume to a square
        # Points in oct space
        src = np.array([
            [0, 0],  # Top left
            [0, self.size_x - 1],  # Top right
            [self.size_z - 1, 0],  # Bottom left
            [self.size_z - 1, self.size_x - 1],
        ])  # Bottom right

        # Respective points in enface space
        dst = np.array([
            (0, 0),  # Top left
            (0, self.size_x - 1),  # Top right
            (self.size_x - 1, 0),  # Bottom left
            (self.size_x - 1, self.size_x - 1),
        ])  # Bottom right

        # Switch from x/y coordinates to row/column coordinates for src and dst
        src = np.flip(src, axis=1)
        dst = np.flip(dst, axis=1)
        # src = src[:, [1, 0]]
        # dst = dst[:, [1, 0]]

        return transform.estimate_transform('affine', src, dst)

    @overload
    def __getitem__(self, index: SupportsIndex) -> EyeBscan:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[EyeBscan]:
        ...

    def __getitem__(
            self, index: Union[SupportsIndex,
                               slice]) -> Union[list[EyeBscan], EyeBscan]:
        """

        Args:
            index:

        Returns:

        """
        # The B-Scan at the given index.
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index = len(self) + index

            if index < len(self):
                try:
                    # Return B-scan with type annotation
                    return self._bscans[index]
                except KeyError:
                    self._bscans[index] = EyeBscan(self, index)
                    return self._bscans[index]
            else:
                raise IndexError()
        else:
            raise TypeError()

    def __len__(self) -> int:
        """The number of B-Scans."""
        return self.shape[0]

    def set_intensity_transform(self, func: Union[str, Callable]) -> None:
        """

        Args:
            func: Either a string specifying a transform from eyepy.core.utils.intensity_transforms or a function

        Returns:

        """
        if isinstance(func, str):
            if func in intensity_transforms:
                self.meta['intensity_transform'] = func
                self.intensity_transform = intensity_transforms[func]
                self._data = None
            elif func == 'custom':
                logger.warning(
                    'Custom intensity transforms can not be loaded currently')
            else:
                logger.warning(
                    f"Provided intensity transform name {func} is not known. Valid names are 'vol' or 'default'. You can also pass your own function."
                )
        elif isinstance(func, Callable):
            self.meta['intensity_transform'] = 'custom'
            self.intensity_transform = func
            self._data = None

    def set_par_algorithm(self, func: Union[str, Callable]) -> None:
        """

        Args:
            func: Either a string specifying a par algorithm from eyepy.core.utils.par_algorithms or a function

        Returns:

        """
        if isinstance(func, str):
            if func in par_algorithms:
                self.meta['par_algorithm'] = func
                self.par_algorithm = par_algorithms[func]
                self._data_par = None
            elif func == 'custom':
                logger.warning(
                    'Custom par algorithms can not be loaded currently')
            else:
                logger.warning(
                    f"Provided par algorithm name {func} is not known. Valid names are 'default'. You can also pass your own function."
                )
        elif isinstance(func, Callable):
            self.meta['par_algorithm'] = 'custom'
            self.par_algorithm = func
            self._data_par = None

    @property
    def data(self) -> np.ndarray:
        """

        Returns:

        """
        if self._data is None:
            self._data = self.intensity_transform(np.copy(self._raw_data))
        return self._data

    @property
    def data_par(self) -> np.ndarray:
        """

        Returns:

        """
        if self._data_par is None:
            self._data_par = self.par_algorithm(np.copy(self._raw_data))
        return self._data_par

    @property
    def shape(self) -> tuple[int, int, int]:
        """

        Returns:

        """
        return self._raw_data.shape

    @shape.setter
    def shape(self, value: tuple[int, int, int]) -> None:
        raise AttributeError(
            'Shape can not be set since it is derived from the data')

    @property
    def scale(self) -> tuple[float, float, float]:
        """

        Returns:

        """
        return self.scale_z, self.scale_y, self.scale_x

    @scale.setter
    def scale(self, value: tuple[float, float, float]) -> None:
        self.scale_z, self.scale_y, self.scale_x = value

    @property
    def size_z(self) -> int:
        """

        Returns:

        """
        return self.shape[0]

    @size_z.setter
    def size_z(self, value: int) -> None:
        raise AttributeError(
            'Size of z axis can not be changed since it is derived from the data'
        )

    @property
    def size_y(self) -> int:
        """

        Returns:

        """
        return self.shape[1]

    @size_y.setter
    def size_y(self, value: int) -> None:
        raise AttributeError(
            'Size of y axis can not be changed since it is derived from the data'
        )

    @property
    def size_x(self) -> int:
        """

        Returns:

        """
        return self.shape[2]

    @size_x.setter
    def size_x(self, value: int) -> None:
        raise AttributeError(
            'Size of x axis can not be changed since it is derived from the data'
        )

    @property
    def scale_z(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_z']

    @scale_z.setter
    def scale_z(self, value: float) -> None:
        self.meta['scale_z'] = value

    @property
    def scale_y(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_y']

    @scale_y.setter
    def scale_y(self, value: float) -> None:
        self.meta['scale_y'] = value

    @property
    def scale_x(self) -> float:
        """

        Returns:

        """
        return self.meta['scale_x']

    @scale_x.setter
    def scale_x(self, value: float) -> None:
        self.meta['scale_x'] = value

    @property
    def scale_unit(self) -> str:
        """

        Returns:

        """
        return self.meta['scale_unit']

    @scale_unit.setter
    def scale_unit(self, value: str) -> None:
        self.meta['scale_unit'] = value

    @property
    def laterality(self) -> str:
        """

        Returns:

        """
        return self.meta['laterality']

    @laterality.setter
    def laterality(self, value: str) -> None:
        self.meta['laterality'] = value

    @property
    def layers(self) -> dict[str, EyeVolumeLayerAnnotation]:
        """

        Returns:

        """
        # Create a dict to access layers by their name. Keys in this dict always
        # reflect the current name attribute of the layers.
        return {layer.name: layer for layer in self._layers}

    @property
    def slabs(self) -> dict[str, EyeVolumeSlabAnnotation]:
        """

        Returns:

        """
        # Create a dict to access slabs by their name
        return {slab.name: slab for slab in self._slabs}

    @property
    def volume_maps(self) -> dict[str, EyeVolumePixelAnnotation]:
        """

        Returns:

        """
        # Create a dict to access volume_maps by their name
        return {vm.name: vm for vm in self._volume_maps}

    def add_pixel_annotation(self,
                             voxel_map: Optional[npt.NDArray[np.bool_]] = None,
                             meta: Optional[dict] = None,
                             **kwargs: Any) -> EyeVolumePixelAnnotation:
        """

        Args:
            voxel_map:
            meta:
            **kwargs:

        Returns:

        """
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        voxel_annotation = EyeVolumePixelAnnotation(self, voxel_map, **meta)
        self._volume_maps.append(voxel_annotation)
        return voxel_annotation

    def remove_pixel_annotation(self, name: str) -> None:
        """

        Args:
            name:

        Returns:

        """
        for i, voxel_map in enumerate(self._volume_maps):
            if voxel_map.name == name:
                self._volume_maps.pop(i)

        # Remove references from B-scans
        for bscan in self:
            if name in bscan.area_maps:
                bscan.area_maps.pop(name)

    def add_layer_annotation(self,
                             height_map: Optional[npt.NDArray[
                                 np.float64]] = None,
                             meta: Optional[dict] = None,
                             **kwargs: Any) -> EyeVolumeLayerAnnotation:
        """

        Args:
            height_map: Height in shape (n_Bscans, Bscan_width) The first index refers to the bottom most B-scan
            meta: name, current_color, and knots
            **kwargs:

        Returns:

        """
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        layer_annotation = EyeVolumeLayerAnnotation(self, height_map, **meta)
        self._layers.append(layer_annotation)
        return layer_annotation

    def remove_layer_annotation(self, name: str) -> None:
        """

        Args:
            name:

        Returns:

        """
        for i, layer in enumerate(self._layers):
            if layer.name == name:
                self._layers.pop(i)

        # Remove references from B-scans
        for bscan in self:
            if name in bscan.layers:
                bscan.layers.pop(name)

    def add_slab_annotation(self,
                            meta: Optional[dict] = None,
                            **kwargs: Any) -> EyeVolumeSlabAnnotation:
        """

        Args:
            meta: Metadata for the slab annotation
            **kwargs: Additional keyword arguments

        Returns:
            EyeVolumeSlabAnnotation: The created slab annotation
        """
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        slab_annotation = EyeVolumeSlabAnnotation(self, **meta)
        self._slabs.append(slab_annotation)
        return slab_annotation

    def remove_slab_annotation(self, name: str) -> None:
        """

        Args:
            name:

        Returns:

        """
        for i, slab in enumerate(self._slabs):
            if slab.name == name:
                self._slabs.pop(i)

        # Remove references from B-scans
        for bscan in self:
            if name in bscan.slabs:
                bscan.slabs.pop(name)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        projections: Union[bool, list[str]] = False,
        slabs: Union[bool, list[str]] = False,
        bscan_region: bool = False,
        bscan_positions: Union[bool, list[int]] = False,
        quantification: Optional[str] = None,
        region: tuple[slice, slice] = np.s_[:, :],
        annotations_only: bool = False,
        projection_kwargs: Optional[dict] = None,
        slab_kwargs: Optional[dict] = None,
        line_kwargs: Optional[dict] = None,
        scalebar: Union[bool, str] = 'botleft',
        scalebar_kwargs: Optional[dict] = None,
        watermark: bool = True,
    ) -> None:
        """Plot an annotated OCT localizer image.

        If the volume does not provide a localizer image an enface projection of the OCT volume is used instead.

        Args:
            ax: Axes to plot on. If not provided plot on the current axes (plt.gca()).
            projections: If `True` plot all projections (default: `False`). If a list of strings is given, plot the projections with the given names. Projections are 2D enface views on oct volume annotations such as drusen.
            slabs: If `True` plot all slab projections (default: `False`). If a list of strings is given, plot the slabs with the given names. Slab projections are 2D enface views on OCTA volume annotations such as NFLVP or SVP.
            bscan_region: If `True` plot the region B-scans are located in (default: `False`)
            bscan_positions: If `True` plot positions of all B-scan (default: `False`). If a list of integers is given, plot the B-scans with the respective indices. Indexing starts at the bottom of the localizer.
            quantification: Name of the OCT volume annotations to plot a quantification for (default: `None`). Quantifications are performed on circular grids.
            region: Region of the localizer to plot (default: `np.s_[...]`)
            annotations_only: If `True` localizer image is not plotted (defaualt: `False`)
            projection_kwargs: Optional keyword arguments for the projection plots. If `None` default values are used (default: `None`). If a dictionary is given, the keys are the projection names and the values are dictionaries of keyword arguments.
            slab_kwargs: Optional keyword arguments for the slab plots. If `None` default values are used (default: `None`). If a dictionary is given, the keys are the slab names and the values are dictionaries of keyword arguments.
            line_kwargs: Optional keyword arguments for customizing the lines to show B-scan region and positions plots. If `None` default values are used which are {"linewidth": 0.2, "linestyle": "-", "color": "green"}
            scalebar: Position of the scalebar, one of "topright", "topleft", "botright", "botleft" or `False` (default: "botleft"). If `True` the scalebar is placed in the bottom left corner. You can custumize the scalebar using the `scalebar_kwargs` argument.
            scalebar_kwargs: Optional keyword arguments for customizing the scalebar. Check the documentation of [plot_scalebar][eyepy.core.plotting.plot_scalebar] for more information.
            watermark: If `True` plot a watermark on the image (default: `True`). When removing the watermark, please consider to cite eyepy in your publication.
        Returns:
            None
        """

        # Complete region index expression
        y_start = region[0].start if region[0].start is not None else 0
        y_stop = region[0].stop if region[
            0].stop is not None else self.localizer.shape[0]
        x_start = region[1].start if region[1].start is not None else 0
        x_stop = region[1].stop if region[
            1].stop is not None else self.localizer.shape[1]

        region = np.s_[y_start:y_stop, x_start:x_stop]

        if ax is None:
            ax = plt.gca()

        if not annotations_only:
            self.localizer.plot(ax=ax,
                                region=region,
                                scalebar=scalebar,
                                scalebar_kwargs=scalebar_kwargs,
                                watermark=watermark)

        if projections is True:
            projections = list(self.volume_maps.keys())
        elif not projections:
            projections = []

        if slabs is True:
            slabs = list(self.slabs.keys())
        elif not slabs:
            slabs = []

        if projection_kwargs is None:
            projection_kwargs = defaultdict(lambda: {})
        for name in projections:
            if name not in projection_kwargs.keys():
                projection_kwargs[name] = {}
            self.volume_maps[name].plot(ax=ax,
                                        region=region,
                                        **projection_kwargs[name])

        if slab_kwargs is None:
            slab_kwargs = defaultdict(lambda: {})
        for name in slabs:
            if name not in slab_kwargs.keys():
                slab_kwargs[name] = {}
            self.slabs[name].plot(ax=ax,
                                  region=region,
                                  transform=True,
                                  **slab_kwargs[name])

        if line_kwargs is None:
            line_kwargs = config.line_kwargs
        else:
            line_kwargs = {**config.line_kwargs, **line_kwargs}

        if bscan_positions:
            self._plot_bscan_positions(
                ax=ax,
                bscan_positions=bscan_positions,
                region=region,
                line_kwargs=line_kwargs,
            )
        if bscan_region:
            self._plot_bscan_region(region=region,
                                    ax=ax,
                                    line_kwargs=line_kwargs)

        if quantification:
            self.volume_maps[quantification].plot_quantification(region=region,
                                                                 ax=ax)

    def _plot_bscan_positions(
        self,
        bscan_positions: Union[bool, list[int]] = True,
        ax: Optional[plt.Axes] = None,
        region: tuple[slice, slice] = np.s_[:, :],
        line_kwargs: Optional[dict] = None,
    ):
        if not bscan_positions:
            bscan_positions = []
        elif bscan_positions is True:
            bscan_positions = list(range(0, len(self)))

        ax = plt.gca() if ax is None else ax
        if line_kwargs is None:
            line_kwargs = {}

        for i in bscan_positions:
            start = self._pos_to_localizer_region(self[i].meta['start_pos'], region)
            end = self._pos_to_localizer_region(self[i].meta['end_pos'], region)

            for pos in [start, end]:
                # Check for both axis if pos is in region
                if not (0 <= pos[0] <= region[1].stop - region[1].start
                        and 0 <= pos[1] <= region[0].stop - region[0].start):
                    logger.warning(
                        'B-scan position can not be plotted because the visualized region does not contain the complete B-scan.'
                    )
                    return

            polygon = patches.Polygon(
                [start, end],
                closed=False,
                fill=False,
                alpha=1,
                antialiased=False,
                rasterized=False,
                snap=False,
                **line_kwargs,
            )
            ax.add_patch(polygon)

    def _pos_to_localizer_region(self, pos: tuple[float, float],
                       region: tuple[slice, slice]) -> tuple[float, float]:

        pos_arr = np.array([pos[0], pos[1]], dtype=float)
        scale = np.array([self.localizer.scale_x, self.localizer.scale_y], dtype=float)

        # Todo: replace with actual localizer FOV
        if self.localizer.scale_unit == '°':
            # 15 degree is half the standard FOV of 30 degree
            # We need the offset since coordinates in degree have the origin in the center
            # of the image
            offset = np.array([15.0, 15.0], dtype=float)
        else:
            offset = np.array([0.0, 0.0], dtype=float)

        region_offset = np.array([region[1].start, region[0].start], dtype=float)

        pos_transformed = ((pos_arr + offset) / scale) - region_offset

        return (float(pos_transformed[0]), float(pos_transformed[1]))

    def _plot_bscan_region(self,
                           region: tuple[slice, slice] = np.s_[:, :],
                           ax: Optional[plt.Axes] = None,
                           line_kwargs: Optional[dict] = None):

        ax = plt.gca() if ax is None else ax
        line_kwargs = {} if line_kwargs is None else line_kwargs

        upper_left = self._pos_to_localizer_region(self[-1].meta['start_pos'], region)
        lower_left = self._pos_to_localizer_region(self[0].meta['start_pos'], region)
        lower_right = self._pos_to_localizer_region(self[0].meta['end_pos'], region)
        upper_right = self._pos_to_localizer_region(self[-1].meta['end_pos'], region)

        for pos in [upper_left, lower_left, lower_right, upper_right]:
            # Check for both axis if pos is in region
            if not (0 <= pos[0] < region[1].stop - region[1].start
                    and 0 <= pos[1] < region[0].stop - region[0].start):
                logger.warning(
                    'B-scan region can not be plotted because the visualized region does not contain the complete B-scan region.'
                )
                return

        polygon = patches.Polygon(
            [upper_left, lower_left, lower_right, upper_right],
            closed=True,
            fill=False,
            alpha=1,
            antialiased=False,
            rasterized=False,
            snap=False,
            **line_kwargs,
        )
        ax.add_patch(polygon)
