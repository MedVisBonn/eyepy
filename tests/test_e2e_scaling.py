import struct
from types import SimpleNamespace

import numpy as np
import pytest

from eyepy.io.he.e2e_format import LateralityEnum
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.he.e2e_reader import E2ESeriesStructure
from eyepy.io.he.e2e_reader import E2ESliceStructure


@pytest.fixture
def series():
    """Two 20-degree B-scans with incomplete filled-A-scan coverage."""
    result = E2ESeriesStructure(1)
    result.folders[TypesEnum.laterality] = [
        SimpleNamespace(data=SimpleNamespace(laterality=LateralityEnum.OD))
    ]
    result.folders[TypesEnum.enface_modality] = [
        SimpleNamespace(data=SimpleNamespace(text=['', 'IR']))
    ]
    result.folders[TypesEnum.image] = [SimpleNamespace(data=SimpleNamespace(
        type=33620481, width=64, height=32, data=np.zeros((32, 64))))]
    settings = bytearray(600)
    struct.pack_into('<d', settings, 457, 7.7)
    for index, y in enumerate((-1, 1)):
        raw = bytearray(438)
        struct.pack_into('<III', raw, 0, 0, 4, 16)
        struct.pack_into('<4f', raw, 12, -10, y, 10, y)
        struct.pack_into('<f', raw, 36, 0.00387)
        struct.pack_into('<IIII', raw, 60, 12, 2, index, 3)
        struct.pack_into('<2f', raw, 76, 0, y)
        struct.pack_into('<d', raw, 136, 0.9 / 0.001043)
        sl = E2ESliceStructure(index * 2)
        sl.folders[TypesEnum.bscanmeta] = [
            SimpleNamespace(get_bytes=lambda raw=raw: bytes(raw))
        ]
        sl.folders[TypesEnum.localizer_settings] = [
            SimpleNamespace(get_bytes=lambda: bytes(settings))
        ]
        sl.folders[TypesEnum.layer_annotation] = []
        sl.get_bscan = lambda: np.zeros((4, 16))
        result.substructure[sl.id] = sl
    return result


def test_default_volume_uses_consistent_mm_and_preserves_alignment(series):
    q = 0.2895099745331661  # R=7.7 mm, F=0.9 D, independently evaluated.
    angular = series.get_bscan_meta()

    with pytest.warns(UserWarning, match='vendor calibration'):
        volume = series.get_volume()

    assert volume.scale_unit == volume.localizer.scale_unit == 'mm'
    assert volume.scale == pytest.approx((2 * q, 0.00387, 20 * q / 16))
    assert volume.localizer.scale_x == pytest.approx(30 * q / 64)
    assert volume.localizer.scale_y == pytest.approx(30 * q / 32)
    assert angular[0]['pos_unit'] == '°'
    assert angular[0]['start_pos'] == (-10, -1)
    assert volume.meta['bscan_meta'][0]['start_pos'] == pytest.approx((5*q, 14*q))
    assert volume.meta['bscan_meta'][0]['pos_unit'] == 'mm'
    # OCT row 0 maps to the final B-scan in the localizer, as before conversion.
    expected = np.array([[5 / 30 * 64, 16 / 30 * 32],
                         [25 / 30 * 64, 14 / 30 * 32]])
    np.testing.assert_allclose(volume.localizer_transform([[0, 0], [15, 1]]), expected)
    info = volume.meta['e2e_metadata']['findings']['scaling_x']
    assert info['status'] == 'approximate'
    assert info['image_width_px'] == 16  # Not the 12 filled A-scans.
    assert info['corneal_radius_mm'] == 7.7
    assert info['focus_diopters'] == pytest.approx(0.9)


@pytest.mark.parametrize(('radius', 'focus', 'expected_q'), [
    (9.0, -1.51, 0.3329369347634005),
    (6.2, -0.49, 0.2560921282567403),
])
def test_optical_conversion_matches_controlled_radius_examples(
        series, monkeypatch, radius, focus, expected_q):
    monkeypatch.setattr(series, 'get_corneal_radius_mm', lambda: radius)
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: focus)
    assert series._mm_per_degree() == pytest.approx(expected_q)


@pytest.mark.parametrize('focus', [None, float('nan')])
def test_missing_optical_settings_use_documented_defaults(series, monkeypatch, focus):
    monkeypatch.setattr(series, 'get_corneal_radius_mm', lambda: None)
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: focus)
    assert series._mm_per_degree() == pytest.approx(1.02302 / (17.21 / 7.7 + 1.247))


@pytest.mark.parametrize('eye,expected_radius', [
    (LateralityEnum.OD, 6.2),
    (LateralityEnum.OS, 9.0),
])
def test_study_radius_fallback_selects_the_matching_eye(series, eye, expected_radius):
    series._type39_raw_cache = bytes(464)
    series.folders[TypesEnum.laterality][0].data.laterality = eye
    series.study = SimpleNamespace(folders={TypesEnum.measurements: [
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OS, corneal_radius_mm=9.0)),
        SimpleNamespace(data=SimpleNamespace(eye_side=LateralityEnum.OD, corneal_radius_mm=6.2)),
    ]})
    assert series.get_corneal_radius_mm() == expected_radius


def test_single_bscan_has_no_measured_through_slice_spacing(series):
    del series.substructure[2]
    with pytest.warns(UserWarning):
        meta = series.get_meta()
    assert meta['scale_z'] == 0


def test_invalid_optical_denominator_is_rejected(series, monkeypatch):
    monkeypatch.setattr(series, '_derive_scan_focus_diopters', lambda: -100)
    with pytest.raises(ValueError, match='non-positive scale'):
        series._mm_per_degree()
