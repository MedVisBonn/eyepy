import numpy as np
import pytest

from eyepy.io.he.e2e_reader import E2ESeriesStructure


class _FakeSlice:

    def __init__(self, data):
        self.data = data

    def get_bscan(self):
        return self.data


def _series(scans):
    series = E2ESeriesStructure(1)
    series.substructure = {
        index * 2: _FakeSlice(scan)
        for index, scan in enumerate(scans)
    }
    series.get_meta = lambda: {
        'bscan_meta': [{
            'size_x': scans[0].shape[1],
            'size_y': scans[0].shape[0]
        }]
    }
    series.get_bscan_meta = lambda: [{
        'size_x': scan.shape[1],
        'size_y': scan.shape[0]
    } for scan in scans]
    return series


def test_get_bscans_reports_slice_with_different_dimensions():
    series = _series([
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
    ])

    with pytest.raises(
            ValueError,
            match=r'B-scan slice 2 has shape \(2, 4\).*different dimensions'):
        series.get_bscans()


def test_get_bscans_keeps_stacking_uniform_slices():
    scans = [
        np.full((2, 3), 1, dtype=np.float32),
        np.full((2, 3), 2, dtype=np.float32),
    ]

    result = _series(scans).get_bscans()

    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result, np.stack(scans))
