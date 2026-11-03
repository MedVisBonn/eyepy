import io
import struct

import numpy as np
import pytest

from eyepy.io import HeXmlReader
from eyepy.io import HeXmlWriter
from eyepy.io.he._xml_tiff import _close_lzw
from eyepy.io.he._xml_tiff import encode_heyex_tiff


def pack_bits(bits):
    bits += '0' * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, 'big')


@pytest.mark.parametrize('data_bits,new_width', [(2286, 10), (7406, 11), (18670, 12)])
def test_terminal_lzw_code_uses_last_data_width(data_bits, new_width):
    # CLEAR followed by literal zero data codes. These bit counts place the
    # last data code exactly at a TIFF early-change dictionary boundary.
    prefix = '100000000' + '0' * data_bits
    encoded = pack_bits(prefix + format(257, f'0{new_width}b'))
    expected = pack_bits(prefix + format(257, f'0{new_width - 1}b'))
    assert _close_lzw(encoded) == expected


def test_terminal_lzw_width_resets_after_clear_and_other_endings_stay_unchanged():
    for bits in ('100000000' + '000000000' + '100000001',
                 '100000000' + '0' * 2286 + format(256, '010b') +
                 '000000000' + '100000001'):
        encoded = pack_bits(bits)
        assert _close_lzw(encoded) == encoded
    with pytest.raises(ValueError, match='no end code'):
        _close_lzw(bytes(1))


def test_terminal_lzw_discards_unused_whole_byte_after_end_code():
    # CLEAR, six literal zero codes and EOI occupy exactly nine bytes.
    expected = pack_bits('100000000' + '0' * 54 + '100000001')
    assert len(expected) == 9
    assert _close_lzw(expected + bytes(1)) == expected


@pytest.mark.parametrize('shape,rows_per_strip', [((5, 1024), 2), ((3, 1536), 1), ((3, 5), 546)])
def test_tiff_profile_and_pixels(shape, rows_per_strip):
    pytest.importorskip('imagecodecs')
    tifffile = pytest.importorskip('tifffile')
    pixels = np.random.default_rng(601).integers(0, 256, shape, dtype=np.uint8)
    original = pixels.copy()
    encoded = encode_heyex_tiff(pixels)
    np.testing.assert_array_equal(pixels, original)
    with tifffile.TiffFile(io.BytesIO(encoded)) as image:
        page = image.pages[0]
        assert list(page.tags) == [page.tags[code] for code in (
            254, 256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 282, 283, 284, 296, 317)]
        assert page.tags[278].value == rows_per_strip
        assert page.rowsperstrip == min(shape[0], rows_per_strip)
        assert page.tags[256].dtype == 3  # SHORT dimensions, not LONG.
        assert page.tags[273].dtype == page.tags[279].dtype == 4
        assert page.tags[282].value == page.tags[283].value == (96 << 24, 1 << 24)
        assert page.dataoffsets[0] == 8
        end = page.dataoffsets[-1] + page.databytecounts[-1]
        assert encoded[end:page.offset] == bytes(end % 2)
        assert struct.unpack_from('<I', encoded, 4)[0] == page.offset
        rgb = page.asarray()
    for channel in range(3):
        np.testing.assert_array_equal(rgb[..., channel], pixels)


@pytest.mark.parametrize('pixels', [np.zeros((2, 3), float),
                                   np.zeros((2, 3, 3), np.uint8),
                                   np.zeros((0, 3), np.uint8),
                                   np.zeros((1, 65536), np.uint8)])
def test_tiff_input_requires_explicit_display_bytes_and_supported_dimensions(pixels):
    with pytest.raises(ValueError):
        encode_heyex_tiff(pixels)


def test_xml_image_replacement_preserves_other_assets_and_xml(tmp_path):
    pytest.importorskip('imagecodecs')
    reader = HeXmlReader('tests/data/filetypes/heyex_xml/test_volume/test_dummy.xml')
    writer = HeXmlWriter(reader)
    before = reader.volume.raw_data
    name = reader.bscan_meta[0]['scan_name']
    changed = before[0].copy()
    changed[2, 3] ^= 255
    writer.replace_image(name, changed)
    output = tmp_path / 'edited'
    writer.write(output)
    loaded = HeXmlReader(output).volume.raw_data
    np.testing.assert_array_equal(loaded[0], changed)
    np.testing.assert_array_equal(loaded[1:], before[1:])
    np.testing.assert_array_equal(reader.volume.raw_data, before)
    assert (output / reader.path.name).read_bytes() == reader.path.read_bytes()
    for asset_name, source in writer.assets.items():
        if asset_name != name:
            assert (output / asset_name).read_bytes() == source.read()
    with pytest.raises(ValueError, match='not referenced'):
        writer.replace_image('unknown.tif', changed)
    with pytest.raises(ValueError, match='dimensions disagree'):
        writer.replace_image(name, changed[:-1])
