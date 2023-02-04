import numpy as np
import pytest

import eyepy as ep


@pytest.fixture(scope="module")
def eyevolume():
    n_bscans = 10
    bscan_height = 50
    bscan_width = 100
    data = np.random.random(
        (n_bscans, bscan_height, bscan_width)).astype(np.float32)
    return ep.EyeVolume(data=data)
