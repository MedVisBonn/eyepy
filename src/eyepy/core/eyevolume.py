from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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

    def save(self, path: Union[str, Path]) -> None:
        """

        Args:
            path:

        Returns:

        """
        # Create temporary folder
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)

            # Save OCT volume as npy and meta as json
            np.save(tmpdirname / 'raw_volume.npy', self._raw_data)
            with open(tmpdirname / 'meta.json', 'w') as meta_file:
                if self.meta['intensity_transform'] == 'custom':
                    warnings.warn(
                        'Custom intensity transforms can not be saved.')
                    self.meta['intensity_transform'] = 'default'
                json.dump(self.meta.as_dict(), meta_file)

            if not len(self._volume_maps) == 0:
                # Save Volume Annotations
                voxels_path = tmpdirname / 'annotations' / 'voxels'
                voxels_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    voxels_path / 'voxel_maps.npy',
                    np.stack([v.data for v in self._volume_maps]),
                )
                with open(voxels_path / 'meta.json', 'w') as meta_file:
                    json.dump([v.meta for v in self._volume_maps], meta_file)

            if not len(self._layers) == 0:
                # Save layer annotations
                layers_path = tmpdirname / 'annotations' / 'layers'
                layers_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    layers_path / 'layer_heights.npy',
                    np.stack([l.data for l in self._layers]),
                )
                with open(layers_path / 'meta.json', 'w') as meta_file:
                    json.dump([l.meta for l in self._layers], meta_file)

            if not len(self._slabs) == 0:
                # Save slab annotations
                slabs_path = tmpdirname / 'annotations' / 'slabs'
                slabs_path.mkdir(exist_ok=True, parents=True)
                with open(slabs_path / 'meta.json', 'w') as meta_file:
                    json.dump([s.meta for s in self._slabs], meta_file)

            # Save Localizer
            localizer_path = tmpdirname / 'localizer'
            localizer_path.mkdir(exist_ok=True, parents=True)
            np.save(localizer_path / 'localizer.npy', self.localizer.data)
            with open(localizer_path / 'meta.json', 'w') as meta_file:
                json.dump(self.localizer.meta.as_dict(), meta_file)

            # Save Localizer Annotations
            if not len(self.localizer._area_maps) == 0:
                pixels_path = localizer_path / 'annotations' / 'pixel'
                pixels_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    pixels_path / 'pixel_maps.npy',
                    np.stack([p.data for p in self.localizer._area_maps]),
                )
                with open(pixels_path / 'meta.json', 'w') as meta_file:
                    json.dump([p.meta for p in self.localizer._area_maps],
                              meta_file)

            # Zip and copy to location
            name = shutil.make_archive(str(path),
                                       'zip',
                                       root_dir=str(tmpdirname))
            # Remove zip extension
            shutil.move(name, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EyeVolume':
        """

        Args:
            path:

        Returns:

        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Load raw volume and meta
            data = np.load(tmpdirname / 'raw_volume.npy')
            with open(tmpdirname / 'meta.json', 'r') as meta_file:
                volume_meta = EyeVolumeMeta.from_dict(json.load(meta_file))

            # Load Volume Annotations
            voxels_path = tmpdirname / 'annotations' / 'voxels'
            if voxels_path.exists():
                voxel_annotations = np.load(voxels_path / 'voxel_maps.npy')
                with open(voxels_path / 'meta.json', 'r') as meta_file:
                    voxels_meta = json.load(meta_file)
            else:
                voxel_annotations = []
                voxels_meta = []

            # Load layers
            layers_path = tmpdirname / 'annotations' / 'layers'
            if layers_path.exists():
                layer_annotations = np.load(layers_path / 'layer_heights.npy')
                with open(layers_path / 'meta.json', 'r') as meta_file:
                    layers_meta = json.load(meta_file)

                # Clean knots
                for i, layer_meta in enumerate(layers_meta):
                    if 'knots' in layer_meta:
                        knots = layer_meta['knots']
                        knots = {int(i): knots[i] for i in knots}
                        layer_meta['knots'] = knots
            else:
                layer_annotations = []
                layers_meta = []

            # Load slabs
            slabs_path = tmpdirname / 'annotations' / 'slabs'
            if slabs_path.exists():
                with open(slabs_path / 'meta.json', 'r') as meta_file:
                    slabs_meta = json.load(meta_file)
            else:
                slabs_meta = []

            # Load Localizer and meta
            localizer_path = tmpdirname / 'localizer'
            localizer_data = np.load(localizer_path / 'localizer.npy')
            with open(localizer_path / 'meta.json', 'r') as meta_file:
                localizer_meta = EyeEnfaceMeta.from_dict(json.load(meta_file))
            localizer = EyeEnface(data=localizer_data, meta=localizer_meta)

            # Load Localizer Annotations
            pixels_path = localizer_path / 'annotations' / 'pixel'
            if pixels_path.exists():
                pixel_annotations = np.load(pixels_path / 'pixel_maps.npy')
                with open(pixels_path / 'meta.json', 'r') as meta_file:
                    pixels_meta = json.load(meta_file)

                for i, pixel_meta in enumerate(pixels_meta):
                    localizer.add_area_annotation(pixel_annotations[i],
                                                  pixel_meta)

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
            scale = np.array([self.localizer.scale_x, self.localizer.scale_y])

            start = self[i].meta['start_pos'] / scale - np.array(
                [region[1].start, region[0].start])
            end = self[i].meta['end_pos'] / scale - np.array(
                [region[1].start, region[0].start])

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

    def _plot_bscan_region(self,
                           region: tuple[slice, slice] = np.s_[:, :],
                           ax: Optional[plt.Axes] = None,
                           line_kwargs: Optional[dict] = None):

        ax = plt.gca() if ax is None else ax
        line_kwargs = {} if line_kwargs is None else line_kwargs

        scale = np.array([self.localizer.scale_x, self.localizer.scale_y])

        upper_left = np.array(self[-1].meta['start_pos']) / scale - np.array(
            [region[1].start, region[0].start])
        lower_left = np.array(self[0].meta['start_pos']) / scale - np.array(
            [region[1].start, region[0].start])
        lower_right = np.array(self[0].meta['end_pos']) / scale - np.array(
            [region[1].start, region[0].start])
        upper_right = np.array(self[-1].meta['end_pos']) / scale - np.array(
            [region[1].start, region[0].start])

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
