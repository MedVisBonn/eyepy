import json
import logging
import shutil
import tempfile
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import typing as npt
from skimage import transform
from skimage.transform._geometric import GeometricTransform

from eyepy import config
from eyepy.core.eyebscan import EyeBscan
from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta, EyeEnfaceMeta, EyeVolumeMeta
from eyepy.core.utils import intensity_transforms

logger = logging.getLogger("eyepy.core.eyevolume")


class LayerKnot(TypedDict):
    knot_pos: Tuple[float, float]
    cp_in_pos: Tuple[float, float]
    cp_out_pos: Tuple[float, float]


class EyeVolumeLayerAnnotation:
    def __init__(
        self,
        volume: "EyeVolume",
        data: Optional[npt.NDArray[np.float32]] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ):
        self.volume = volume
        if data is None:
            self.data = np.full((volume.size_z, volume.size_x), np.nan)
        else:
            self.data = data

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        if "knots" not in self.meta:
            self.meta["knots"] = defaultdict(lambda: [])
        elif type(self.meta["knots"]) is dict:
            self.meta["knots"] = defaultdict(lambda: [], self.meta["knots"])

        if "name" not in self.meta:
            self.meta["name"] = "Layer Annotation"

        self.meta["current_color"] = config.layer_colors[self.name]

    @property
    def name(self):
        return self.meta["name"]

    @name.setter
    def name(self, value):
        self.meta["name"] = value

    @property
    def knots(self):
        return self.meta["knots"]

    def layer_indices(self):
        layer = self.data
        nan_indices = np.isnan(layer)
        col_indices = np.arange(len(layer))[~nan_indices]
        row_indices = np.rint(layer).astype(int)[~nan_indices]

        return (row_indices, col_indices)


class EyeVolumeVoxelAnnotation:
    def __init__(
        self,
        volume: "EyeVolume",
        data: Optional[npt.NDArray[bool]] = None,
        meta: Optional[dict] = None,
        radii=(1.5, 2.5),
        n_sectors=(1, 4),
        offsets=(0, 45),
        center=None,
        **kwargs,
    ):
        self.volume = volume

        if data is None:
            self.data = np.full(self.volume.shape, fill_value=False, dtype=bool)
        else:
            self.data = data

        self._masks = None
        self._quantification = None

        if meta is None:
            self.meta = kwargs
        else:
            self.meta = meta
            self.meta.update(**kwargs)

        self.meta.update(
            **{
                "radii": radii,
                "n_sectors": n_sectors,
                "offsets": offsets,
                "center": center,
            }
        )

        if "name" not in self.meta:
            self.meta["name"] = "Voxel Annotation"

    @property
    def name(self):
        return self.meta["name"]

    @name.setter
    def name(self, value):
        self.meta["name"] = value

    def _reset(self):
        self._masks = None
        self._quantification = None

    @property
    def radii(self):
        return self.meta["radii"]

    @radii.setter
    def radii(self, value):
        self._reset()
        self.meta["radii"] = value

    @property
    def n_sectors(self):
        return self.meta["n_sectors"]

    @n_sectors.setter
    def n_sectors(self, value):
        self._reset()
        self.meta["n_sectors"] = value

    @property
    def offsets(self):
        return self.meta["offsets"]

    @offsets.setter
    def offsets(self, value):
        self._reset()
        self.meta["offsets"] = value

    @property
    def center(self):
        return self.meta["center"]

    @center.setter
    def center(self, value):
        self._reset()
        self.meta["center"] = value

    @property
    def projection(self):
        return np.flip(np.nansum(self.data, axis=1), axis=0)

    @property
    def enface(self):
        return transform.warp(
            self.projection,
            self.volume.localizer_transform.inverse,
            output_shape=(
                self.volume.localizer.size_y,
                self.volume.localizer.size_x,
            ),
            order=0,
        )

    def plot(
        self,
        ax=None,
        region=np.s_[...],
        cmap="Reds",
        vmin=None,
        vmax=None,
        cbar=True,
        alpha=1,
    ):
        enface_projection = self.enface

        if ax is None:
            ax = plt.gca()

        if vmin is None:
            vmin = 1
        if vmax is None:
            vmax = max([enface_projection.max(), vmin])

        visible = np.zeros(enface_projection[region].shape)
        visible[
            np.logical_and(
                vmin <= enface_projection[region], enface_projection[region] <= vmax
            )
        ] = 1

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                cax=cax,
            )

        ax.imshow(
            enface_projection[region],
            alpha=visible[region] * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    @property
    def masks(self):
        from eyepy.quantification.utils.grids import grid

        if self._masks is None:
            self._masks = grid(
                mask_shape=self.volume.localizer.shape,
                radii=self.radii,
                laterality=self.volume.laterality,
                n_sectors=self.n_sectors,
                offsets=self.offsets,
                radii_scale=self.volume.scale_x,
                center=self.center,
            )

        return self._masks

    @property
    def quantification(self):
        if self._quantification is None:
            self._quantification = self._quantify()

        return self._quantification

    def _quantify(self):
        enface_voxel_size_ym3 = (
            self.volume.localizer.scale_x
            * 1e3
            * self.volume.localizer.scale_y
            * 1e3
            * self.volume.scale_y
            * 1e3
        )
        oct_voxel_size_ym3 = (
            self.volume.scale_x
            * 1e3
            * self.volume.scale_z
            * 1e3
            * self.volume.scale_y
            * 1e3
        )

        enface_projection = self.enface

        results = {}
        for name, mask in self.masks.items():
            results[f"{name} [mm³]"] = (
                (enface_projection * mask).sum() * enface_voxel_size_ym3 / 1e9
            )

        results["Total [mm³]"] = enface_projection.sum() * enface_voxel_size_ym3 / 1e9
        results["Total [OCT voxels]"] = self.projection.sum()
        results["OCT Voxel Size [µm³]"] = oct_voxel_size_ym3
        results["Laterality"] = self.volume.laterality
        return results

    def plot_quantification(
        self,
        ax=None,
        region=np.s_[...],
        alpha=0.5,
        vmin=None,
        vmax=None,
        cbar=True,
        cmap="YlOrRd",
    ):

        if ax is None:
            ax = plt.gca()

        mask_img = np.zeros(self.volume.localizer.shape, dtype=float)[region]
        visible = np.zeros_like(mask_img)
        for mask_name in self.masks.keys():
            mask_img += (
                self.masks[mask_name][region]
                * self.quantification[mask_name + " [mm³]"]
            )
            visible += self.masks[mask_name][region]

        if vmin is None:
            vmin = mask_img[visible.astype(int)].min()
        if vmax is None:
            vmax = max([mask_img.max(), vmin])

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(
                cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                cax=cax,
            )

        ax.imshow(
            mask_img,
            alpha=visible * alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )


class EyeVolume:
    def __init__(
        self,
        data: npt.NDArray[np.float32],
        meta: EyeVolumeMeta = None,
        ascan_maps=None,
        localizer: "EyeEnface" = None,
        transformation: GeometricTransform = None,
    ):
        self._raw_data = data
        self._data = None

        self._bscans = {}

        if meta is None:
            self.meta = self._default_meta(self._raw_data)
        else:
            self.meta = meta
        if not "intensity_transform" in self.meta:
            self.meta["intensity_transform"] = "default"

        self.set_intensity_transform(self.meta["intensity_transform"])

        self._layers = []
        self._volume_maps = []

        if ascan_maps is None:
            self.ascan_maps = {}
        else:
            self.ascan_maps = ascan_maps

        if transformation is None:
            self.localizer_transform = self._estimate_transform()
        else:
            self.localizer_transform = transformation

        if localizer is None:
            self.localizer = self._default_localizer(self.data)
        else:
            self.localizer = localizer

    def save(self, path):
        # Create temporary folder
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)

            # Save OCT volume as npy and meta as json
            np.save(tmpdirname / "raw_volume.npy", self._raw_data)
            with open(tmpdirname / "meta.json", "w") as meta_file:
                if self.meta["intensity_transform"] == "custom":
                    warnings.warn("Custom intensity transforms can not be saved.")
                    self.meta["intensity_transform"] = "default"
                json.dump(self.meta.as_dict(), meta_file)

            if not len(self._volume_maps) == 0:
                # Save Volume Annotations
                voxels_path = tmpdirname / "annotations" / "voxels"
                voxels_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    voxels_path / "voxel_maps.npy",
                    np.stack([v.data for v in self._volume_maps]),
                )
                with open(voxels_path / "meta.json", "w") as meta_file:
                    json.dump([v.meta for v in self._volume_maps], meta_file)

            if not len(self._layers) == 0:
                layers_path = tmpdirname / "annotations" / "layers"
                layers_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    layers_path / "layer_heights.npy",
                    np.stack([l.data for l in self._layers]),
                )
                with open(layers_path / "meta.json", "w") as meta_file:
                    json.dump([l.meta for l in self._layers], meta_file)

            # Save Localizer
            localizer_path = tmpdirname / "localizer"
            localizer_path.mkdir(exist_ok=True, parents=True)
            np.save(localizer_path / "localizer.npy", self.localizer.data)
            with open(localizer_path / "meta.json", "w") as meta_file:
                json.dump(self.localizer.meta.as_dict(), meta_file)

            # Save Localizer Annotations
            if not len(self.localizer._area_maps) == 0:
                pixels_path = localizer_path / "annotations" / "pixel"
                pixels_path.mkdir(exist_ok=True, parents=True)
                np.save(
                    pixels_path / "pixel_maps.npy",
                    np.stack([p.data for p in self.localizer._area_maps]),
                )
                with open(pixels_path / "meta.json", "w") as meta_file:
                    json.dump([p.meta for p in self.localizer._area_maps], meta_file)

            # Zip and copy to location
            name = shutil.make_archive(Path(path).stem, "zip", tmpdirname)
            shutil.move(name, path)

    @classmethod
    def load(cls, path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Load raw volume and meta
            data = np.load(tmpdirname / "raw_volume.npy")
            with open(tmpdirname / "meta.json", "r") as meta_file:
                volume_meta = EyeVolumeMeta.from_dict(json.load(meta_file))

            # Load Volume Annotations
            voxels_path = tmpdirname / "annotations" / "voxels"
            if voxels_path.exists():
                voxel_annotations = np.load(voxels_path / "voxel_maps.npy")
                with open(voxels_path / "meta.json", "r") as meta_file:
                    voxels_meta = json.load(meta_file)

            # Load layers
            layers_path = tmpdirname / "annotations" / "layers"
            if layers_path.exists():
                layer_annotations = np.load(layers_path / "layer_heights.npy")
                with open(layers_path / "meta.json", "r") as meta_file:
                    layers_meta = json.load(meta_file)

            # Load Localizer and meta
            localizer_path = tmpdirname / "localizer"
            localizer_data = np.load(localizer_path / "localizer.npy")
            with open(localizer_path / "meta.json", "r") as meta_file:
                localizer_meta = EyeEnfaceMeta.from_dict(json.load(meta_file))
            localizer = EyeEnface(data=localizer_data, meta=localizer_meta)

            # Load Localizer Annotations
            pixels_path = localizer_path / "annotations" / "pixel"
            if pixels_path.exists():
                pixel_annotations = np.load(pixels_path / "pixel_maps.npy")
                with open(pixels_path / "meta.json", "r") as meta_file:
                    pixels_meta = json.load(meta_file)

                for i, pixel_meta in enumerate(pixels_meta):
                    localizer.add_area_annotation(pixel_annotations[i], pixel_meta)

            from eyepy.io.utils import _compute_localizer_oct_transform

            transformation = _compute_localizer_oct_transform(
                volume_meta, localizer_meta, data.shape
            )

            ev = cls(
                data=data,
                meta=volume_meta,
                localizer=localizer,
                transformation=transformation,
            )
            if layers_path.exists():
                for i, layer_meta in enumerate(layers_meta):
                    if "knots" in layer_meta:
                        knots = layer_meta["knots"]
                        knots = {int(i): knots[i] for i in knots}
                        layer_meta["knots"] = knots
                    ev.add_layer_annotation(layer_annotations[i], layer_meta)

            if voxels_path.exists():
                for i, voxel_meta in enumerate(voxels_meta):
                    ev.add_voxel_annotation(voxel_annotations[i], voxel_meta)
        return ev

    def save_annotations(self, path):
        pass

    def load_annotations(self, path):
        pass

    def _default_meta(self, volume):
        bscan_meta = [
            EyeBscanMeta(
                start_pos=(0, i), end_pos=((volume.shape[2] - 1), i), pos_unit="pixel"
            )
            for i in range(volume.shape[0] - 1, -1, -1)
        ]
        meta = EyeVolumeMeta(
            scale_x=1,
            scale_y=1,
            scale_z=1,
            scale_unit="pixel",
            intensity_transform="default",
            bscan_meta=bscan_meta,
        )
        return meta

    def _default_localizer(self, data):
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
                scale_x=self.scale_x, scale_y=self.scale_x, scale_unit="mm"
            ),
        )
        return localizer

    def _estimate_transform(self):
        """Compute a transform to map a 2D projection of the volume to a square"""
        # Points in oct space
        src = np.array(
            [
                [0, 0],  # Top left
                [0, self.size_x - 1],  # Top right
                [self.size_z - 1, 0],  # Bottom left
                [self.size_z - 1, self.size_x - 1],
            ]
        )  # Bottom right

        # Respective points in enface space
        dst = np.array(
            [
                (0, 0),  # Top left
                (0, self.size_x - 1),  # Top right
                (self.size_x - 1, 0),  # Bottom left
                (self.size_x - 1, self.size_x - 1),
            ]
        )  # Bottom right

        # Switch from x/y coordinates to row/column coordinates
        src = src[:, [1, 0]]
        dst = dst[:, [1, 0]]
        return transform.estimate_transform("affine", src, dst)

    def __getitem__(self, index) -> Union[EyeBscan, List[EyeBscan]]:
        """The B-Scan at the given index."""
        if type(index) == slice:
            return [self[i] for i in range(*index.indices(len(self)))]

        if index < 0:
            index = len(self) + index

        if index < len(self):
            try:
                return self._bscans[index]
            except KeyError:
                self._bscans[index] = EyeBscan(self, index)
                return self._bscans[index]
        else:
            raise IndexError()

    def __len__(self):
        """The number of B-Scans."""
        return self.shape[0]

    def set_intensity_transform(self, func: Union[str, Callable]):
        """

        Args:
            func: Either a string specifying a transform from eyepy.core.utils.intensity_transforms or a function

        Returns:

        """
        if type(func) is str:
            if func in intensity_transforms:
                self.meta["intensity_transform"] = func
                self.intensity_transform = intensity_transforms[func]
                self._data = None
            elif func == "custom":
                logger.warning(
                    "Custom intensity transforms can not be loaded currently"
                )
            else:
                logger.warning(
                    f"Provided intensity transform name {func} is not know. Valid names are 'vol' or 'default'. You can also pass your own function."
                )
        else:
            self.meta["intensity_transform"] = "custom"
            self.intensity_transform = func
            self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self.intensity_transform(np.copy(self._raw_data))
        return self._data

    @property
    def shape(self):
        return self._raw_data.shape

    @property
    def scale(self):
        return self.scale_z, self.scale_y, self.scale_x

    @property
    def size_z(self):
        return self.shape[0]

    @property
    def size_y(self):
        return self.shape[1]

    @property
    def size_x(self):
        return self.shape[2]

    @property
    def scale_z(self):
        return self.meta["scale_z"]

    @property
    def scale_y(self):
        return self.meta["scale_y"]

    @property
    def scale_x(self):
        return self.meta["scale_x"]

    @property
    def laterality(self):
        return self.meta["laterality"]

    @property
    def layers(self):
        # Create a dict to access layers by their name
        return {layer.name: layer for layer in self._layers}

    @property
    def volume_maps(self):
        # Create a dict to access volume_maps by their name
        return {vm.name: vm for vm in self._volume_maps}

    def add_voxel_annotation(self, voxel_map=None, meta=None, **kwargs):
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        voxel_annotation = EyeVolumeVoxelAnnotation(self, voxel_map, **meta)
        self._volume_maps.append(voxel_annotation)
        return voxel_annotation

    def delete_voxel_annotations(self, name):
        for i, voxel_map in enumerate(self._volume_maps):
            if voxel_map.name == name:
                self._volume_maps.pop(i)

        # Remove references from B-scans
        for bscan in self:
            if name in bscan.area_maps:
                bscan.area_maps.pop(name)

    def add_layer_annotation(self, height_map=None, meta=None, **kwargs):
        if meta is None:
            meta = {}
        meta.update(**kwargs)
        layer_annotation = EyeVolumeLayerAnnotation(self, height_map, **meta)
        self._layers.append(layer_annotation)
        return layer_annotation

    def delete_layer_annotation(self, name):
        for i, layer in enumerate(self._layers):
            if layer.name == name:
                self._layers.pop(i)

        # Remove references from B-scans
        for bscan in self:
            if name in bscan.layers:
                bscan.layers.pop(name)

    def plot(
        self,
        ax=None,
        localizer=True,
        projections=False,
        bscan_region=False,
        bscan_positions=None,
        quantification=None,
        region=np.s_[...],
        projection_kwargs=None,
        line_kwargs=None,
    ):
        if ax is None:
            ax = plt.gca()

        if localizer:
            self.localizer.plot(ax=ax, region=region)

        if projections is True:
            projections = list(self.volume_maps.keys())
        elif not projections:
            projections = []

        if projection_kwargs is None:
            projection_kwargs = defaultdict(lambda: {})
        for name in projections:
            if not name in projection_kwargs.keys():
                projection_kwargs[name] = {}
            self.volume_maps[name].plot(ax=ax, region=region, **projection_kwargs[name])

        if line_kwargs is None:
            line_kwargs = config.line_kwargs
        else:
            line_kwargs = {**config.line_kwargs, **line_kwargs}

        if bscan_positions is not None:
            self._plot_bscan_positions(
                ax=ax,
                bscan_positions=bscan_positions,
                region=region,
                line_kwargs=line_kwargs,
            )
        if bscan_region:
            self._plot_bscan_region(region=region, ax=ax, line_kwargs=line_kwargs)

        if quantification:
            self.volume_maps[quantification].plot_quantification(region=region, ax=ax)

    def plot_bscan_ticks(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.yticks()

    def _plot_bscan_positions(
        self, bscan_positions="all", ax=None, region=np.s_[...], line_kwargs=None
    ):
        if not bscan_positions:
            bscan_positions = []
        elif bscan_positions == "all" or bscan_positions is True:
            bscan_positions = range(0, len(self))

        for i in bscan_positions:
            scale = np.array([self.localizer.scale_x, self.localizer.scale_y])

            start = self[i].meta["start_pos"] / scale
            end = self[i].meta["end_pos"] / scale

            # x = [start[0], end[0]]
            # y = [start[1], end[1]]
            # ax.plot(x, y, **line_kwargs)
            polygon = patches.Polygon(
                np.array([start, end]),
                closed=False,
                fill=False,
                alpha=1,
                antialiased=False,
                rasterized=False,
                snap=False,
                **line_kwargs,
            )
            ax.add_patch(polygon)

    def _plot_bscan_region(self, region=np.s_[...], ax=None, line_kwargs=None):
        if ax is None:
            ax = plt.gca()

        scale = np.array([self.localizer.scale_x, self.localizer.scale_y])

        upper_left = self[-1].meta["start_pos"] / scale
        lower_left = self[0].meta["start_pos"] / scale
        lower_right = self[0].meta["end_pos"] / scale
        upper_right = self[-1].meta["end_pos"] / scale

        polygon = patches.Polygon(
            np.array([upper_left, lower_left, lower_right, upper_right]),
            closed=True,
            fill=False,
            alpha=1,
            antialiased=False,
            rasterized=False,
            snap=False,
            **line_kwargs,
        )
        ax.add_patch(polygon)
