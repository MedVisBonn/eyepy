import struct
from types import SimpleNamespace
from uuid import UUID

import pytest

from eyepy.io.he.e2e_reader import E2EAcquisitionKey
from eyepy.io.he.e2e_reader import E2ESeriesStructure

PROGRESSION = UUID('12345678-1234-5678-1234-567812345678')
TICKS = 116444736000000000


def folder(payload):
    return SimpleNamespace(get_bytes=lambda: payload)


def series(own_ticks=TICKS, reference_ticks=None, progression=PROGRESSION):
    result = E2ESeriesStructure(1)
    if own_ticks is not None:
        result.folders[10005] = [folder(progression.bytes_le + struct.pack('<Q', own_ticks))]
    if reference_ticks is not None:
        result.folders[10025] = [folder(bytes(8) + progression.bytes_le +
                                       struct.pack('<Q', reference_ticks) + bytes(68))]
    return result


def test_reference_lookup_preserves_sub_microsecond_acquisition_identity():
    baseline = series(TICKS, TICKS)
    follow_up = series(TICKS + 1, TICKS)
    assert baseline.get_acquisition_key() != follow_up.get_acquisition_key()
    assert follow_up.get_reference_key() == baseline.get_acquisition_key()
    assert follow_up.resolve_reference([follow_up, baseline]) is baseline
    assert follow_up.get_acquisition_key().as_dict() == dict(
        progression_id=str(PROGRESSION), windows_ticks=TICKS + 1)


def test_self_reference_and_missing_reference_record_return_the_own_series():
    for source in (series(TICKS, TICKS), series(TICKS), series(None)):
        assert source.resolve_reference() is source
    assert series(None).get_acquisition_key() is None
    assert series(TICKS).get_reference_key() is None


def test_external_reference_is_required_instead_of_silently_using_follow_up():
    source = series(TICKS + 10, TICKS)
    with pytest.raises(ValueError, match='required but was not supplied'):
        source.resolve_reference()
    wrong_progression = series(TICKS, progression=UUID(int=1))
    with pytest.raises(ValueError, match='required but was not supplied'):
        source.resolve_reference([wrong_progression])


def test_multiple_exports_with_the_same_reference_key_are_ambiguous():
    source = series(TICKS + 10, TICKS)
    first, second = series(), series()
    with pytest.raises(ValueError, match='Multiple supplied'):
        source.resolve_reference([first, second])
    assert source.resolve_reference([first, first]) is first


@pytest.mark.parametrize('kind,offset', [(10005, 0), (10025, 8)])
def test_truncated_and_ambiguous_keys_are_rejected(kind, offset):
    source = series(None)
    source.folders[kind] = [folder(bytes(offset + 23))]
    method = source.get_acquisition_key if kind == 10005 else source.get_reference_key
    with pytest.raises(ValueError, match='Truncated'):
        method()
    source.folders[kind] = [folder(bytes(offset + 24))] * 2
    with pytest.raises(ValueError, match='Ambiguous'):
        method()


def test_zero_key_is_unspecified_and_keys_are_hashable():
    source = series(None)
    source.folders[10025] = [folder(bytes(100))]
    assert source.get_reference_key() is None
    assert source.resolve_reference() is source
    assert {E2EAcquisitionKey(PROGRESSION, TICKS), series().get_acquisition_key()} == {
        E2EAcquisitionKey(PROGRESSION, TICKS)}
