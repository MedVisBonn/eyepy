import numpy as np
import pytest

from eyepy.core.grids import grid
from eyepy.quant.grid_presets import ETDRS_9


def test_grid_preset_names_match_mask_count():
    output = grid(
        mask_shape=(31, 31),
        radii=ETDRS_9.radii,
        laterality='OD',
        n_sectors=ETDRS_9.n_sectors,
        offsets=ETDRS_9.offsets,
        grid_preset=ETDRS_9,
    )
    assert set(output.keys()) == set(ETDRS_9.region_names)


def test_grid_preset_od_and_os_share_keys():
    output_od = grid(
        mask_shape=(31, 31),
        radii=ETDRS_9.radii,
        laterality='OD',
        n_sectors=ETDRS_9.n_sectors,
        offsets=ETDRS_9.offsets,
        grid_preset=ETDRS_9,
    )
    output_os = grid(
        mask_shape=(31, 31),
        radii=ETDRS_9.radii,
        laterality='OS',
        n_sectors=ETDRS_9.n_sectors,
        offsets=ETDRS_9.offsets,
        grid_preset=ETDRS_9,
    )

    assert output_od.keys() == output_os.keys()
    for key in output_od:
        assert np.all(output_od[key] == np.flip(output_os[key], axis=1))


def test_grid_custom_names_override():
    custom_names = list(ETDRS_9.region_names)
    output = grid(
        mask_shape=(31, 31),
        radii=ETDRS_9.radii,
        laterality='OD',
        n_sectors=ETDRS_9.n_sectors,
        offsets=ETDRS_9.offsets,
        names=custom_names,
    )
    assert list(output.keys()) == custom_names


def test_grid_invalid_names_length_raises():
    with pytest.raises(ValueError, match='Expected 9 region names'):
        grid(
            mask_shape=(31, 31),
            radii=ETDRS_9.radii,
            laterality='OD',
            n_sectors=ETDRS_9.n_sectors,
            offsets=ETDRS_9.offsets,
            names=['OnlyOne'],
        )


def test_grid_index_naming_default():
    output = grid(
        mask_shape=(15, 15),
        radii=(1.5, 2.5),
        laterality='OD',
        n_sectors=(1, 4),
        offsets=(0, 45),
    )
    assert list(output.keys()) == [
        'Radius: 0.0-1.5 Sector: 0',
        'Radius: 1.5-2.5 Sector: 0',
        'Radius: 1.5-2.5 Sector: 1',
        'Radius: 1.5-2.5 Sector: 2',
        'Radius: 1.5-2.5 Sector: 3',
    ]
