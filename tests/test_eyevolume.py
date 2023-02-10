import collections

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.utils import DynamicDefaultDict


def test_eyevolume_attributes_shape(eyevolume):
    assert eyevolume.size_z == 10
    assert eyevolume.size_x == 100
    assert eyevolume.size_y == 50

    # Sizes / shape can not be set since they are derived from the data
    with pytest.raises(AttributeError):
        eyevolume.size_z = 5
    with pytest.raises(AttributeError):
        eyevolume.size_x = 5
    with pytest.raises(AttributeError):
        eyevolume.size_y = 5
    with pytest.raises(AttributeError):
        eyevolume.shape = (5, 5, 5)

    assert eyevolume.shape == (eyevolume.size_z, eyevolume.size_y,
                               eyevolume.size_x)


def test_eyevolume_attributes_scale(eyevolume):
    # The scale is np.nan if not specified
    assert eyevolume.scale_z == 1.0
    assert eyevolume.scale_x == 1.0
    assert eyevolume.scale_y == 1.0
    assert eyevolume.scale_unit == "pixel"

    # Test setting scaling information
    eyevolume.scale_z = 0.5
    eyevolume.scale_x = 0.5
    eyevolume.scale_y = 0.5
    eyevolume.scale_unit = "mm"

    assert eyevolume.scale_z == 0.5
    assert eyevolume.scale_x == 0.5
    assert eyevolume.scale_y == 0.5
    assert eyevolume.scale_unit == "mm"
    assert eyevolume.scale == (eyevolume.scale_z, eyevolume.scale_y,
                               eyevolume.scale_x)


def test_eyevolume_attributes(eyevolume):
    # Test whether localizer was created (projection from volume)
    assert type(eyevolume.localizer) == ep.EyeEnface
    # Test whether layer_annotations were created
    assert type(eyevolume.layers) == dict
    # Test whether volume annotations were created
    assert type(eyevolume.volume_maps) == dict


def test_eyevolume_indexing_and_iteration(eyevolume):
    # Length of eyevolume is the number of B-scans
    assert len(eyevolume) == 10
    # Index of first B-scan is 0
    assert eyevolume[0].index == 0
    # Index of last B-scan is the number of B-scans - 1
    assert eyevolume[-1].index == 9
    # Slicing returns a list (of B-scans)
    assert len(eyevolume[:5]) == 5
    # Looping over eyevolume returns B-scans
    for bscan in eyevolume:
        assert isinstance(bscan, ep.EyeBscan)


def test_eyevolume_meta_exists(eyevolume):
    assert isinstance(eyevolume.meta, ep.EyeVolumeMeta)
    assert isinstance(eyevolume[0].meta, ep.EyeBscanMeta)
    assert isinstance(eyevolume.localizer.meta, ep.EyeEnfaceMeta)


# Set layers on eyevolume
def test_set_layers_on_eyevolume(eyevolume):
    eyevolume.add_layer_annotation(np.full(
        (eyevolume.size_z, eyevolume.size_x), 25),
                                   name="test_layer")
    assert type(eyevolume.layers["test_layer"]) == ep.EyeVolumeLayerAnnotation
    assert np.all(eyevolume[5].layers["test_layer"].data == 25)
    assert eyevolume.layers["test_layer"].data[5, 10] == 25


# Set layers on eyebscan
def test_set_layers_on_eyebscan(eyevolume):
    eyevolume.add_layer_annotation(name="test_layer_2")
    assert "test_layer_2" in eyevolume.layers.keys()

    bscan = eyevolume[5]
    assert (type(bscan.layers["test_layer_2"].eyevolumelayerannotation) ==
            ep.EyeVolumeLayerAnnotation)
    # Check that all heights are initially nan
    assert np.sum(~np.isnan(bscan.layers["test_layer_2"].data)) == 0

    # Check that correct values are set
    bscan.layers["test_layer_2"].data = np.full((eyevolume.size_x, ), 240)
    assert np.all(bscan.layers["test_layer_2"].data == 240)
    assert np.nansum(
        eyevolume.layers["test_layer_2"].data) == eyevolume.size_x * 240
    assert np.all(eyevolume.layers["test_layer_2"].data[-(5 + 1)] == 240)


# Test Bscan iteration
def test_bscan_iteration(eyevolume):
    bscans = [b for b in eyevolume]
    bscans_2 = [eyevolume[i] for i in range(len(eyevolume))]
    assert bscans == bscans_2


# Delete layers
def test_delete_layers(eyevolume):
    eyevolume.add_layer_annotation(name="delete_layer")
    assert "delete_layer" in eyevolume.layers

    eyevolume[2].layers["delete_layer"].data = 20
    assert "delete_layer" in eyevolume[2].layers

    eyevolume.remove_layer_annotation("delete_layer")

    assert "delete_layer" not in eyevolume.layers
    # Test for references in the B-scans
    assert "delete_layer" not in eyevolume[2].layers


def test_delete_voxel_annotation(eyevolume):
    eyevolume.add_voxel_annotation(name="delete_volume")
    assert "delete_volume" in eyevolume.volume_maps

    eyevolume[2].area_maps["delete_volume"][:5, :5] = 20
    assert "delete_volume" in eyevolume[2].area_maps

    eyevolume.remove_voxel_annotations("delete_volume")

    assert "delete_volume" not in eyevolume.volume_maps
    # Test for references in the B-scans
    assert "delete_volume" not in eyevolume[2].area_maps


#def test_save_load(eyevolume, tmp_path):
#    # Save
#    eyevolume.save(tmp_path / "test.eye")
#    # Load
#    eyevolume2 = ep.EyeVolume.load(tmp_path / "test.eye")
#    # Test whether loaded eyevolume is the same as the original
#    assert len(eyevolume.meta) == len(eyevolume2.meta)
#    assert len(eyevolume.layers) == len(eyevolume2.layers)
#    assert len(eyevolume.volume_maps) == len(eyevolume2.volume_maps)
