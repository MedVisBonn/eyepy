import numpy as np

from eyepy.quant.thickness import thickness_from_layer_pair
from eyepy.quant.thickness import thickness_from_voxel_annotation


def test_thickness_from_layer_pair():
    top = np.array([[10.0, 10.0], [10.0, 10.0]])
    bottom = np.array([[15.0, 20.0], [12.0, 18.0]])
    scale_y = 2.0

    thickness = thickness_from_layer_pair(top, bottom, scale_y)

    expected = np.array([[10.0, 20.0], [4.0, 16.0]])
    np.testing.assert_allclose(thickness, expected)


def test_thickness_from_layer_pair_nan_propagation():
    top = np.array([[10.0, np.nan]])
    bottom = np.array([[15.0, 18.0]])
    scale_y = 1.0

    thickness = thickness_from_layer_pair(top, bottom, scale_y)

    assert thickness[0, 0] == 5.0
    assert np.isnan(thickness[0, 1])


def test_thickness_from_layer_pair_shape_mismatch():
    top = np.zeros((2, 3))
    bottom = np.zeros((2, 4))
    try:
        thickness_from_layer_pair(top, bottom, 1.0)
    except ValueError as exc:
        assert 'shape mismatch' in str(exc)
    else:
        raise AssertionError('Expected ValueError for shape mismatch')


def test_thickness_from_voxel_annotation():
    mask = np.zeros((2, 5, 3), dtype=bool)
    mask[0, 2:4, 0] = True
    mask[0, 1:4, 1] = True
    scale_y = 0.5

    thickness = thickness_from_voxel_annotation(mask, scale_y)

    assert thickness.shape == (2, 3)
    assert thickness[0, 0] == 1.0
    assert thickness[0, 1] == 1.5
    assert thickness[0, 2] == 0.0
    assert thickness[1].sum() == 0.0


def test_masked_mean_and_median():
    from eyepy.quant.metrics import masked_mean
    from eyepy.quant.metrics import masked_median

    data = np.array([1.0, np.nan, 3.0, 4.0])
    mask = np.array([True, True, True, False])

    assert masked_mean(data, mask) == 2.0
    assert masked_median(data, mask) == 2.0
    assert np.isnan(masked_mean(data, np.zeros(4, dtype=bool)))
