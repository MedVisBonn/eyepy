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
    assert eyevolume.scale_unit == 'pixel'

    # Test setting scaling information
    eyevolume.scale_z = 0.5
    eyevolume.scale_x = 0.5
    eyevolume.scale_y = 0.5
    eyevolume.scale_unit = 'mm'

    assert eyevolume.scale_z == 0.5
    assert eyevolume.scale_x == 0.5
    assert eyevolume.scale_y == 0.5
    assert eyevolume.scale_unit == 'mm'
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
                                   name='test_layer')
    assert type(eyevolume.layers['test_layer']) == ep.EyeVolumeLayerAnnotation
    assert np.all(eyevolume[5].layers['test_layer'].data == 25)
    assert eyevolume.layers['test_layer'].data[5, 10] == 25


# Set layers on eyebscan
def test_set_layers_on_eyebscan(eyevolume):
    eyevolume.add_layer_annotation(name='test_layer_2')
    assert 'test_layer_2' in eyevolume.layers.keys()

    bscan = eyevolume[5]
    assert (type(bscan.layers['test_layer_2'].eyevolumelayerannotation) ==
            ep.EyeVolumeLayerAnnotation)
    # Check that all heights are initially nan
    assert np.sum(~np.isnan(bscan.layers['test_layer_2'].data)) == 0

    # Check that correct values are set
    bscan.layers['test_layer_2'].data = np.full((eyevolume.size_x, ), 240)
    assert np.all(bscan.layers['test_layer_2'].data == 240)
    assert np.nansum(
        eyevolume.layers['test_layer_2'].data) == eyevolume.size_x * 240
    assert np.all(eyevolume.layers['test_layer_2'].data[5] == 240)


# Test Bscan iteration
def test_bscan_iteration(eyevolume):
    bscans = [b for b in eyevolume]
    bscans_2 = [eyevolume[i] for i in range(len(eyevolume))]
    assert bscans == bscans_2


# Delete layers
def test_delete_layers(eyevolume):
    eyevolume.add_layer_annotation(name='delete_layer')
    assert 'delete_layer' in eyevolume.layers

    eyevolume[2].layers['delete_layer'].data = 20
    assert 'delete_layer' in eyevolume[2].layers

    eyevolume.remove_layer_annotation('delete_layer')

    assert 'delete_layer' not in eyevolume.layers
    # Test for references in the B-scans
    assert 'delete_layer' not in eyevolume[2].layers


def test_delete_voxel_annotation(eyevolume):
    eyevolume.add_pixel_annotation(name='delete_volume')
    assert 'delete_volume' in eyevolume.volume_maps

    eyevolume[2].area_maps['delete_volume'][:5, :5] = 20
    assert 'delete_volume' in eyevolume[2].area_maps

    eyevolume.remove_pixel_annotation('delete_volume')

    assert 'delete_volume' not in eyevolume.volume_maps
    # Test for references in the B-scans
    assert 'delete_volume' not in eyevolume[2].area_maps


def test_save_load(eyevolume, tmp_path):
    """Test that save and load preserve all data exactly.

    Verifies:
    - Raw volume data
    - Metadata (scale, laterality, bscan_meta)
    - Layer annotations
    - Volume (voxel) annotations
    - Slab annotations
    - Area (localizer pixel) annotations
    - Localizer image
    - Localizer metadata
    - Localizer transform
    """


    # Save
    eyevolume.save(tmp_path / 'test.eye', compress=True)
    # Load
    eyevolume2 = ep.EyeVolume.load(tmp_path / 'test.eye')

    # 1. Verify raw volume data is identical
    np.testing.assert_array_equal(
        eyevolume._raw_data,
        eyevolume2._raw_data,
        err_msg='Raw volume data does not match'
    )

    # 2. Verify metadata
    assert eyevolume.meta == eyevolume2.meta, 'Metadata does not match'
    assert eyevolume.scale == eyevolume2.scale, 'Scale does not match'
    assert eyevolume.laterality == eyevolume2.laterality, 'Laterality does not match'
    assert eyevolume.scale_unit == eyevolume2.scale_unit, 'Scale unit does not match'

    # 3. Verify layer annotations
    assert len(eyevolume.layers) == len(eyevolume2.layers), \
        f'Number of layers mismatch: {len(eyevolume.layers)} vs {len(eyevolume2.layers)}'

    for layer_name in eyevolume.layers:
        assert layer_name in eyevolume2.layers, \
            f"Layer '{layer_name}' not found in loaded volume"
        original_layer = eyevolume.layers[layer_name]
        loaded_layer = eyevolume2.layers[layer_name]

        np.testing.assert_array_equal(
            original_layer.data,
            loaded_layer.data,
            err_msg=f"Layer '{layer_name}' data does not match"
        )
        assert original_layer.meta == loaded_layer.meta, \
            f"Layer '{layer_name}' metadata does not match"

    # 4. Verify volume (voxel) annotations / volume maps
    assert len(eyevolume.volume_maps) == len(eyevolume2.volume_maps), \
        f'Number of volume maps mismatch: {len(eyevolume.volume_maps)} vs {len(eyevolume2.volume_maps)}'

    for vmap_name in eyevolume.volume_maps:
        assert vmap_name in eyevolume2.volume_maps, \
            f"Volume map '{vmap_name}' not found in loaded volume"
        original_vmap = eyevolume.volume_maps[vmap_name]
        loaded_vmap = eyevolume2.volume_maps[vmap_name]

        np.testing.assert_array_equal(
            original_vmap.data,
            loaded_vmap.data,
            err_msg=f"Volume map '{vmap_name}' data does not match"
        )
        # Note: JSON serialization converts tuples to lists, so we normalize for comparison
        assert set(original_vmap.meta.keys()) == set(loaded_vmap.meta.keys()), \
            f"Volume map '{vmap_name}' metadata keys do not match"
        for key in original_vmap.meta:
            orig_val = original_vmap.meta[key]
            loaded_val = loaded_vmap.meta[key]
            # Normalize: convert both to lists for comparison (JSON converts tuples->lists)
            if isinstance(orig_val, (tuple, list)):
                orig_val = list(orig_val)
            if isinstance(loaded_val, (tuple, list)):
                loaded_val = list(loaded_val)
            assert orig_val == loaded_val, \
                f"Volume map '{vmap_name}' metadata['{key}'] does not match: {original_vmap.meta[key]} != {loaded_vmap.meta[key]}"

    # 5. Verify slab annotations
    assert len(eyevolume.slabs) == len(eyevolume2.slabs), \
        f'Number of slabs mismatch: {len(eyevolume.slabs)} vs {len(eyevolume2.slabs)}'

    for slab_name in eyevolume.slabs:
        assert slab_name in eyevolume2.slabs, \
            f"Slab '{slab_name}' not found in loaded volume"
        original_slab = eyevolume.slabs[slab_name]
        loaded_slab = eyevolume2.slabs[slab_name]
        assert original_slab.meta == loaded_slab.meta, \
            f"Slab '{slab_name}' metadata does not match"

    # 6. Verify area (localizer pixel) annotations
    original_area_maps = eyevolume.localizer._area_maps
    loaded_area_maps = eyevolume2.localizer._area_maps

    assert len(original_area_maps) == len(loaded_area_maps), \
        f'Number of area maps mismatch: {len(original_area_maps)} vs {len(loaded_area_maps)}'

    for i, (orig_amap, loaded_amap) in enumerate(zip(original_area_maps, loaded_area_maps)):
        np.testing.assert_array_equal(
            orig_amap.data,
            loaded_amap.data,
            err_msg=f'Area map {i} data does not match'
        )
        assert orig_amap.meta == loaded_amap.meta, \
            f'Area map {i} metadata does not match'

    # 7. Verify localizer image data
    np.testing.assert_array_equal(
        eyevolume.localizer.data,
        eyevolume2.localizer.data,
        err_msg='Localizer image data does not match'
    )

    # 8. Verify localizer metadata
    assert eyevolume.localizer.meta == eyevolume2.localizer.meta, \
        'Localizer metadata does not match'
    assert eyevolume.localizer.scale_x == eyevolume2.localizer.scale_x, \
        'Localizer scale_x does not match'
    assert eyevolume.localizer.scale_y == eyevolume2.localizer.scale_y, \
        'Localizer scale_y does not match'
    assert eyevolume.localizer.laterality == eyevolume2.localizer.laterality, \
        'Localizer laterality does not match'

    # 9. Verify localizer transform parameters
    if eyevolume.localizer_transform is not None:
        assert eyevolume2.localizer_transform is not None, \
            'Localizer transform was not loaded'
        np.testing.assert_array_almost_equal(
            eyevolume.localizer_transform.params,
            eyevolume2.localizer_transform.params,
            err_msg='Localizer transform parameters do not match'
        )
