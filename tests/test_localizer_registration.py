from types import SimpleNamespace

import numpy as np
import pytest
from skimage.transform import AffineTransform

from eyepy import EyeBscanMeta
from eyepy import EyeEnface
from eyepy import EyeEnfaceMeta
from eyepy import EyeVolume
from eyepy import EyeVolumeMeta
from eyepy.core.heidelberg import HeidelbergBscanTransform
from eyepy.core.heidelberg import HeidelbergLocalizerTransform
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.he.e2e_reader import E2ESeriesStructure
from eyepy.io.he.e2e_reader import E2ESliceStructure
from eyepy.io.he.e2e_reader import HeE2eReader


def localizer(matrix, unit='pixel', scale=1):
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    return EyeEnface(pixels,
                     EyeEnfaceMeta(scale_x=scale, scale_y=scale, scale_unit=unit),
                     spatial_transform=HeidelbergLocalizerTransform(matrix))


def test_localizer_sampling_truncates_and_preserves_original_pixels():
    image = EyeEnface(np.array([[0, 255], [100, 200]], np.uint8),
                     EyeEnfaceMeta(scale_x=1, scale_y=1, scale_unit='pixel'),
                     spatial_transform=HeidelbergLocalizerTransform(
                         [[1, 0, .5], [0, 1, -.5]]))
    original = image.raw_data
    assert image.data[0, 0] == 127
    assert image.data[0, 1] == 0  # Upper half-pixel boundary is excluded.
    assert image.data[1, 0] == 138
    np.testing.assert_array_equal(image.raw_data, original)


def test_centered_rotation_maps_image_and_positions_in_the_same_direction():
    transform = HeidelbergLocalizerTransform([[0, -1, 1], [1, 0, -2]])
    source = np.zeros((4, 6), np.uint8)
    source[1, 4] = 200
    assert transform(source)[2, 4] == 200
    np.testing.assert_allclose(transform.source_to_output([[4, 1]], source.shape),
                               [[4, 2]])


def test_identity_localizer_transform_is_exact():
    image = localizer([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(image.data, image.raw_data)


@pytest.mark.parametrize('matrix', [
    [[1, 0, 0]],
    [[1, 0, float('nan')], [0, 1, 0]],
    [[1, 2, 0], [2, 4, 0]],
])
def test_invalid_localizer_affines_are_rejected(matrix):
    with pytest.raises(ValueError):
        HeidelbergLocalizerTransform(matrix)


@pytest.mark.parametrize(('unit', 'position', 'expected'), [
    ('°', (0, 0), (-2, 0)),
    ('mm', (4, 4), (2, 4)),
])
def test_endpoint_registration_preserves_original_metadata(unit, position, expected):
    image = localizer([[1, 0, 2], [0, 1, 0]], unit=unit)
    meta = EyeBscanMeta(start_pos=position, end_pos=position, pos_unit=unit)
    result, = E2ESeriesStructure(1)._registered_bscan_positions([meta], image)
    np.testing.assert_allclose(result['start_pos'], expected)
    assert result['raw_start_pos'] == position
    assert meta['start_pos'] == position
    assert 'raw_start_pos' not in meta


def test_localizer_registration_and_raw_pixels_survive_eye_save_load(tmp_path):
    image = localizer([[1, 0, 2], [0, 1, 0]])
    meta = EyeVolumeMeta(scale_x=1, scale_y=1, scale_z=1, scale_unit='pixel',
                         bscan_meta=[EyeBscanMeta(start_pos=(0, i), end_pos=(3, i),
                                                  pos_unit='pixel') for i in range(2)])
    volume = EyeVolume(np.ones((2, 3, 4), np.float32), meta,
                       image, AffineTransform())
    path = tmp_path / 'localizer.eye'
    volume.save(path)
    restored = EyeVolume.load(path)
    np.testing.assert_array_equal(restored.localizer.raw_data, image.raw_data)
    np.testing.assert_array_equal(restored.localizer.data, image.data)
    np.testing.assert_array_equal(restored.localizer.spatial_transform.matrix,
                                  image.spatial_transform.matrix)


def test_materializing_an_enface_transform_does_not_register_twice():
    image = localizer([[1, 0, 2], [0, 1, 0]])
    transformed = image.transform(np.eye(3))
    assert transformed.spatial_transform is None
    np.testing.assert_array_equal(transformed.raw_data, image.registered_data)
    np.testing.assert_array_equal(transformed.data, image.data)


@pytest.mark.parametrize('slice_local', [False, True])
def test_reader_raw_localizer_view_does_not_inherit_cached_registration(slice_local):
    series = E2ESeriesStructure(1)
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image_folders = series.folders
    if slice_local:
        sl = E2ESliceStructure(0)
        series.substructure[sl.id] = sl
        image_folders = sl.folders
    image_folders[TypesEnum.image] = [SimpleNamespace(data=SimpleNamespace(
        data=pixels, height=8, width=8, type=33620481))]
    series.folders[TypesEnum.slodata] = [SimpleNamespace(data=SimpleNamespace(
        transform=[1, 0, 2, 0, 1, 0]))]
    series._localizer_meta = EyeEnfaceMeta(scale_x=1, scale_y=1, scale_unit='pixel')
    registered = series.get_localizer(registered=True)
    original = series.get_localizer(registered=False)
    assert registered.spatial_transform is not None
    assert original.spatial_transform is None
    assert 'spatial_transform' not in series._localizer_meta
    np.testing.assert_array_equal(original.data, pixels)


@pytest.mark.parametrize('attribute', ['volume', 'volumes'])
def test_reader_preserves_invalid_localizer_registration_error(monkeypatch, attribute):
    series = E2ESeriesStructure(1)
    series.folders[TypesEnum.slodata] = [SimpleNamespace(data=SimpleNamespace(
        transform=[1, 2, 0, 2, 4, 0]))]
    monkeypatch.setattr(series, 'get_volume', series.get_localizer_transform)
    reader = HeE2eReader.__new__(HeE2eReader)
    monkeypatch.setattr(HeE2eReader, 'series', property(lambda self: [series]))
    with pytest.raises(ValueError, match='localizer affine is singular'):
        getattr(reader, attribute)


@pytest.mark.parametrize('apply_registration', [False, True])
def test_reader_keeps_bscan_shift_separate_and_updates_overlay_geometry(
        monkeypatch, apply_registration):
    series = E2ESeriesStructure(1)
    image = localizer([[1, 0, 2], [0, 1, 0]], unit='mm')
    bscan_meta = [EyeBscanMeta(start_pos=(4, 4 + i), end_pos=(9, 4 + i),
                              pos_unit='mm', scan_pattern=3) for i in range(2)]
    meta = EyeVolumeMeta(scale_x=1, scale_y=1, scale_z=1,
                         scale_unit='mm', bscan_meta=bscan_meta)
    bscan_transform = HeidelbergBscanTransform(
        [1, 1], [[[1, 0, .5], [0, 1, 0]]] * 2)
    monkeypatch.setattr(series, 'get_meta', lambda: meta)
    monkeypatch.setattr(series, 'get_bscans', lambda: np.ones((2, 3, 6), np.float32))
    monkeypatch.setattr(series, 'get_spatial_transform',
                        lambda: bscan_transform if apply_registration else None)
    def get_localizer(registered=False):
        assert registered == apply_registration
        if registered:
            return image
        return EyeEnface(image.raw_data, EyeEnfaceMeta(
            scale_x=1, scale_y=1, scale_unit='mm'))
    monkeypatch.setattr(series, 'get_localizer', get_localizer)
    monkeypatch.setattr(series, 'get_layers', lambda: {})
    monkeypatch.setattr(series, 'get_layer_manual_maps', lambda: {})
    volume = series.get_volume()
    if not apply_registration:
        assert volume.meta['bscan_meta'][0]['start_pos'] == (4, 4)
        assert 'raw_start_pos' not in volume.meta['bscan_meta'][0]
        assert volume.localizer.spatial_transform is None
        assert volume.scale_x == 1
        return
    assert volume.meta['bscan_meta'][0]['start_pos'] == (2, 4)
    assert volume.meta['bscan_meta'][0]['raw_start_pos'] == (4, 4)
    assert meta['bscan_meta'][0]['start_pos'] == (4, 4)
    assert volume.scale_x == pytest.approx(5 / 6)
    np.testing.assert_allclose(volume.localizer_transform([[0, 0], [5, 1]]),
                               [[2, 5], [7, 4]])
