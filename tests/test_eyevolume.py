import collections

import numpy as np
import pytest

import eyepy as ep
from eyepy.core.utils import DynamicDefaultDict


@pytest.fixture(scope="module")
def eyevolume():
    return ep.EyeVolume(data=np.random.random((10, 50, 100)))


def test_eyevolume_attributes(eyevolume):
    assert eyevolume.size_z == 10
    assert eyevolume.size_x == 100
    assert eyevolume.size_y == 50


def test_eyevolume_iteration(eyevolume):
    assert len(eyevolume) == 10
    assert eyevolume[-1].index == 9
    assert eyevolume[5].index == 5
    assert eyevolume[5] is eyevolume[5]
    for i, bscan in enumerate(eyevolume):
        assert bscan.index == i
        assert bscan.index < len(eyevolume)


def test_eyevolume_attributes(eyevolume):
    assert type(eyevolume.localizer) == ep.EyeEnface
    assert type(eyevolume.layers) == dict
    assert type(eyevolume.volume_maps) == dict
    assert eyevolume.shape == (eyevolume.size_z, eyevolume.size_y, eyevolume.size_x)
    assert eyevolume.scale == (eyevolume.scale_z, eyevolume.scale_y, eyevolume.scale_x)


def test_eyevolume_meta_exists(eyevolume):
    assert type(eyevolume.meta) == ep.EyeVolumeMeta
    assert type(eyevolume[0].meta) == ep.EyeBscanMeta
    assert type(eyevolume.localizer.meta) == ep.EyeEnfaceMeta


# Set layers on eyevolume
def test_set_layers_on_eyevolume(eyevolume):
    eyevolume.add_layer_annotation(
        np.full((eyevolume.size_z, eyevolume.size_x), 250), name="test_layer"
    )
    assert type(eyevolume.layers["test_layer"]) == ep.EyeVolumeLayerAnnotation
    assert np.all(eyevolume[5].layers["test_layer"].data == 250)
    assert eyevolume.layers["test_layer"].data[5, 10] == 250


# Set layers on eyebscan
def test_set_layers_on_eyebscan(eyevolume):
    eyevolume.add_layer_annotation(name="test_layer_2")
    assert "test_layer_2" in eyevolume.layers.keys()

    bscan = eyevolume[5]
    assert (
        type(bscan.layers["test_layer_2"].eyevolumelayerannotation)
        == ep.EyeVolumeLayerAnnotation
    )
    # Check that all heights are initially nan
    assert np.sum(~np.isnan(bscan.layers["test_layer_2"].data)) == 0

    # Check that correct values are set
    bscan.layers["test_layer_2"].data = np.full((eyevolume.size_x,), 240)
    assert np.all(bscan.layers["test_layer_2"].data == 240)
    assert np.nansum(eyevolume.layers["test_layer_2"].data) == eyevolume.size_x * 240
    assert np.all(eyevolume.layers["test_layer_2"].data[-(5 + 1)] == 240)
