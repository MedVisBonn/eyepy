from io import BytesIO
import struct
from types import SimpleNamespace

import pytest

from eyepy.io.he.e2e_reader import E2EFolder


def make_folder(payload, typ=10004, following=b''):
    header = struct.pack('<12sIIIIIiiiiHHII', b'MDbData', 0, 0, 0,
                         len(payload), 0, 1, 2, 3, 0, 1, 0, typ, 0)
    reader = SimpleNamespace(file_object=BytesIO(bytes(88) + header + payload + following))
    return E2EFolder(1, 2, 3, 0, 0, 88, typ, len(payload), 1, reader)


def test_greedy_metadata_stops_before_the_next_container():
    folder = make_folder(bytes(438), following=b'NEIGHBOR' * 100)
    assert len(folder.data.rest) == 278
    assert folder.header.size == 438
    assert folder.get_bytes() == bytes(438)


def test_empty_payload_remains_cached_after_file_closes():
    folder = make_folder(b'', typ=123456)
    assert folder.data == b''
    folder.file_object.close()
    assert folder.data == b''


def test_directory_and_container_sizes_must_agree():
    folder = make_folder(bytes(438))
    folder.size += 1
    with pytest.raises(ValueError, match='sizes disagree'):
        _ = folder.data


def test_truncated_container_header_is_reported():
    folder = make_folder(bytes(438))
    folder.file_object.truncate(88 + 59)
    with pytest.raises(EOFError, match='header'):
        _ = folder.data


@pytest.mark.parametrize('read', ['data', 'get_bytes'])
def test_truncated_payload_is_reported_and_not_cached(read):
    folder = make_folder(bytes(438))
    folder.file_object.truncate(88 + 60 + 437)
    with pytest.raises(EOFError, match='payload'):
        if read == 'data':
            _ = folder.data
        else:
            folder.get_bytes()
    assert folder._bytes is None
