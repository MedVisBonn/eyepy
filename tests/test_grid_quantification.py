import numpy as np

from eyepy.quant.grid import quantize_on_grid


def test_quantize_on_grid_mean():
    data_map = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    masks = {
        'zone_a': np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=float),
        'zone_b': np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=float),
    }

    results = quantize_on_grid(
        data_map,
        masks,
        aggregator='mean',
        unit='mm',
    )

    assert np.isclose(results['zone_a [mm]'], 7.0 / 3.0)
    assert np.isclose(results['zone_b [mm]'], 38.0 / 6.0)


def test_quantize_on_grid_sum_with_scale_factor():
    data_map = np.ones((3, 3))
    masks = {'all': np.ones((3, 3))}

    results = quantize_on_grid(
        data_map,
        masks,
        aggregator='sum',
        unit='mm³',
        scale_factor=2.0,
    )

    assert results['all [mm³]'] == 18.0


def test_quantize_on_grid_median():
    data_map = np.array([1.0, 100.0, 3.0])
    masks = {'line': np.array([1, 1, 1], dtype=float)}

    results = quantize_on_grid(
        data_map,
        masks,
        aggregator='median',
        unit='µm',
    )

    assert results['line [µm]'] == 3.0
