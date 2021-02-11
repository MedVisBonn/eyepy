# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod

from skimage import transform

from eyepy.utils.masks import create_region_masks, create_region_shape_primitives

logger = logging.getLogger(__name__)


class EyeQuantifier(ABC):
    @abstractmethod
    def regions(self, oct_obj):
        raise NotImplementedError()

    @abstractmethod
    def quantify(self, oct_obj):
        raise NotImplementedError()

    @abstractmethod
    def plot_primitives(self, oct_obj):
        raise NotImplementedError()


class DefaultEyeQuantifier(EyeQuantifier):
    def __init__(self):
        self._regions = None
        self.radii = [0.8, 1.8]

    def regions(self, oct_obj):
        """A dict `region_name:mask` for regions in the OCT."""
        if self._regions is None:
            radii_enface_space = [r / oct_obj.ScaleXSlo for r in self.radii]
            masks = create_region_masks(
                (oct_obj.SizeXSlo, oct_obj.SizeYSlo),
                radii=radii_enface_space,
                n_sectors=[4, 1, 1],
                rotation=[45, 0, 0],
                center=oct_obj.fovea_pos,
                smooth_edges=False,
                ring_sectors=True,
                add_circle_masks=True,
            )

            region_names = {
                "OS": ["Superior", "Nasal", "Inferior", "Temporal"],
                "OD": ["Superior", "Temporal", "Inferior", "Nasal"],
            }
            names = region_names[oct_obj.ScanPosition]
            names = [f"{n} 0.8-1.8" for n in names]
            [names.append(f"Radius-{r}") for r in self.radii]
            self._regions = {name: mask for name, mask in zip(names, masks)}
        return self._regions

    def plot_primitives(self, oct_obj):
        radii_enface_space = [r / oct_obj.ScaleXSlo for r in self.radii]
        return create_region_shape_primitives(
            (oct_obj.SizeXSlo, oct_obj.SizeYSlo),
            radii=radii_enface_space,
            n_sectors=[4, 1, 1],
            rotation=[45, 0, 0],
            center=oct_obj.fovea_pos,
        )

    def quantify(self, oct_obj, space="enface"):
        """Quantify the defined regions.

        B-Scan Voxels are often stretched perpendicular to the B-Scan direction.

        Parameters
        ----------
        oct_obj :
        space : Either `enface` or `oct`. This specifies whether to quantify in
        the enface space by warping the oct drusen projection on the enface or
        in oct space, by warping the quantification masks on the oct drusen
        projection. `enface` is a little faster since only one image and not all
        masks have to be warped.

        Returns
        -------
        """
        oct_voxel_size_µm3 = (
            oct_obj.ScaleX * 1e3 * oct_obj.Distance * 1e3 * oct_obj.ScaleZ * 1e3
        )
        enface_voxel_size_µm3 = (
            oct_obj.ScaleXSlo * 1e3 * oct_obj.ScaleYSlo * 1e3 * oct_obj.ScaleZ * 1e3
        )

        if space == "enface":
            drusen_enface = oct_obj.drusen_enface
            voxel_size_µm3 = enface_voxel_size_µm3
        elif space == "oct":
            drusen_enface = oct_obj.drusen_projection
            voxel_size_µm3 = oct_voxel_size_µm3
        else:
            raise ValueError("Choose 'oct' or 'enface' for the space parameter")

        masks = self.regions(oct_obj)
        results = {}
        for name, mask in masks.items():
            # The voxel size in enface space not in the measurement (oct) space
            if space == "oct":
                mask = transform.warp(mask, oct_obj.tform_localizer_to_oct)[
                    : oct_obj.NumBScans, : oct_obj.SizeX
                ]
            results[f"{name} [mm³]"] = (
                (drusen_enface * mask).sum() * voxel_size_µm3 / 1e9
            )

        results["Total [mm³]"] = drusen_enface.sum() * voxel_size_µm3 / 1e9
        results["Total [OCT voxels]"] = oct_obj.drusen_projection.sum()
        results["OCT Voxel Size [µm³]"] = oct_voxel_size_µm3
        # results["Enface Voxel Size [µm³]"] = enface_voxel_size_µm3
        return results
