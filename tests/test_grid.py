import numpy as np

from eyepy.core.grids import grid


def test_grid_1_circle():
    output_OD = grid(
        mask_shape=(5, 5),
        radii=2,
        laterality="OD",
    )
    output_OS = grid(
        mask_shape=(5, 5),
        radii=2,
        laterality="OS",
    )

    assert len(output_OS) == 1
    assert output_OS.keys() == output_OD.keys()
    for key in output_OS.keys():
        assert np.all(output_OD[key] == np.flip(output_OS[key], axis=1))


def test_grid_2_rings():
    output_OD = grid(
        mask_shape=(7, 7), radii=(1, 3), laterality="OD", n_sectors=1, offsets=0
    )
    output_OS = grid(
        mask_shape=(7, 7), radii=(1, 3), laterality="OS", n_sectors=1, offsets=0
    )

    assert len(output_OS) == 2
    assert output_OS.keys() == output_OD.keys()
    for key in output_OS.keys():
        assert np.all(output_OD[key] == np.flip(output_OS[key], axis=1))


def test_grid_1_circle_with_4_sectors():
    output_OD = grid(mask_shape=(7, 7), radii=(3,), laterality="OD", n_sectors=(4,))
    output_OS = grid(mask_shape=(7, 7), radii=(3,), laterality="OS", n_sectors=(4,))

    assert len(output_OS) == 4
    assert output_OS.keys() == output_OD.keys()
    for key in output_OS.keys():
        assert np.all(output_OD[key] == np.flip(output_OS[key], axis=1))


def test_grid_1_circle_with_4_sectors_with_offset_45():
    output_OD = grid(
        mask_shape=(8, 8), radii=(3,), laterality="OD", n_sectors=(4,), offsets=(45,)
    )
    output_OS = grid(
        mask_shape=(8, 8), radii=(3,), laterality="OS", n_sectors=(4,), offsets=(45,)
    )

    assert len(output_OS) == 4
    assert output_OS.keys() == output_OD.keys()
    for key in output_OS.keys():
        assert np.all(output_OD[key] == np.flip(output_OS[key], axis=1))


def test_grid():
    output_OD = grid(
        mask_shape=(15, 15),
        radii=(2, 5),
        laterality="OD",
        n_sectors=(1, 4),
        offsets=(120,),
    )
    output_OS = grid(
        mask_shape=(15, 15),
        radii=(2, 5),
        laterality="OS",
        n_sectors=(1, 4),
        offsets=(120,),
    )

    assert len(output_OS) == 5
    assert output_OS.keys() == output_OD.keys()
    for key in output_OS.keys():
        assert np.all(output_OD[key] == np.flip(output_OS[key], axis=1))
