import collections

import eyepy as ep
import numpy as np
import pytest


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
    assert type(eyevolume.layers) == collections.defaultdict
    assert eyevolume.shape == (eyevolume.size_z, eyevolume.size_y, eyevolume.size_x)
    assert eyevolume.scale == (eyevolume.scale_z, eyevolume.scale_y, eyevolume.scale_x)


def test_eyevolume_meta_exists(eyevolume):
    assert type(eyevolume.meta) == ep.EyeVolumeMeta
    assert type(eyevolume[0].meta) == ep.EyeBscanMeta
    assert type(eyevolume.localizer.meta) == ep.EyeEnfaceMeta


# Set layers on eyevolume
def test_set_layers_on_eyevolume(eyevolume):
    eyevolume.layers["test_layer"] = ep.EyeVolumeLayerAnnotation(
        eyevolume, np.full((eyevolume.size_z, eyevolume.size_x), 250)
    )
    assert type(eyevolume.layers["test_layer"]) == ep.EyeVolumeLayerAnnotation
    assert np.all(eyevolume[5].layers["test_layer"] == 250)
    assert eyevolume.layers["test_layer"].data[5, 10] == 250


# Set layers on eyebscan
def test_set_layers_on_eyebscan(eyevolume):
    bscan = eyevolume[5]
    bscan.layers["bscan_layer"] = np.full((eyevolume.size_x,), 240)

    assert "bscan_layer" in eyevolume.layers.keys()
    assert eyevolume.layers["bscan_layer"].data[5].shape == (100,)
    assert np.sum(~np.isnan(eyevolume.layers["bscan_layer"].data)) == 100
    assert eyevolume.layers["bscan_layer"].data[-(5 + 1)][0] == 240
    assert np.all(np.isnan(eyevolume.layers["bscan_layer"].data[0]))
