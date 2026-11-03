"""Encode the RGB TIFF profile observed in paired HEYEX XML exports.

This serializes rendered pixels. It does not choose intensity, geometry
or calibration, and does not establish fidelity of an entire XML export.
"""

from __future__ import annotations

import struct

import numpy as np


def _close_lzw(encoded: bytes) -> bytes:
    """Use HEYEX's terminal code width at dictionary-growth boundaries.

    The data codes use TIFF's early-change LZW convention. At the end of
    a strip, HEYEX writes EOI with the last data code's width, even when
    decoding that last code grows the dictionary width. Current codecs
    grow the width for EOI as well. Read equal-width groups to locate
    EOI and resets without constructing the decoded dictionary.
    """
    data = np.frombuffer(encoded, np.uint8)
    position, width, next_code = 0, 9, 258
    first, grew = True, False
    while position + width <= len(data) * 8:
        available = (((1 << width) - 1 - next_code) + int(first)
                     if width < 12 else (len(data) * 8 - position) // width)
        count = min(available, (len(data) * 8 - position) // width)
        if count <= 0:
            raise ValueError('Invalid TIFF LZW dictionary state')
        bits = position + width * np.arange(count, dtype=np.int64)
        indices = bits // 8
        words = ((data[indices].astype(np.uint32) << 16) |
                 (data[np.minimum(indices + 1, len(data) - 1)].astype(np.uint32) << 8) |
                 data[np.minimum(indices + 2, len(data) - 1)])
        codes = (words >> (24 - width - bits % 8)) & ((1 << width) - 1)
        stops = np.flatnonzero((codes == 256) | (codes == 257))
        used = int(stops[0]) if stops.size else count
        if stops.size and codes[used] == 257:
            start = position + used * width
            if grew and used == 0:
                terminal_width = width - 1
                size = (start + terminal_width + 7) // 8
                tail = 8 * size - start
                value = int.from_bytes(encoded[:size], 'big')
                value = (value >> tail) << tail
                value |= 257 << (tail - terminal_width)
                return value.to_bytes(size, 'big')
            # Some codec versions append a whole zero byte when EOI ends
            # on a byte boundary. It is not part of the HEYEX strip.
            return encoded[:(start + width + 7) // 8]
        position += (used + int(bool(stops.size))) * width
        if stops.size:
            width, next_code, first, grew = 9, 258, True, False
        else:
            next_code += used - int(first)
            first, grew = False, False
            if width < 12 and next_code == (1 << width) - 1:
                width += 1
                grew = True
    raise ValueError('TIFF LZW strip has no end code')


def encode_heyex_tiff(display: np.ndarray) -> bytes:
    """Encode rendered uint8 grayscale pixels with the observed HEYEX layout.

    The profile uses little-endian classic TIFF, RGB grayscale triplets,
    8 KiB nominal strips, horizontal prediction and LZW. IFD order,
    scalar types, 96 dpi rationals and zero alignment bytes are explicit.
    Requires the optional ``eyepy[tiff]`` dependency. Exact reproduction
    has only been verified against the paired exports documented in
    ``docs/development/heyex-export-fidelity.md``.
    """
    display = np.asarray(display)
    if display.ndim != 2 or display.dtype != np.uint8:
        raise ValueError('HEYEX TIFF encoding requires a 2D uint8 display image')
    height, width = display.shape
    if not (0 < height <= 65535 and 0 < width <= 65535):
        raise ValueError('Image dimensions exceed the observed HEYEX TIFF profile')
    try:
        import imagecodecs
    except ImportError as error:
        raise ImportError('HEYEX TIFF encoding requires eyepy[tiff]') from error

    rows_per_strip = max(1, 8192 // (width * 3))
    payload, strip_offsets, strip_counts = bytearray(), [], []
    for row in range(0, height, rows_per_strip):
        pixels = display[row:row + rows_per_strip]
        # TIFF horizontal prediction restarts at each row and subtracts the
        # preceding pixel's corresponding channel, with uint8 wraparound.
        delta = np.empty_like(pixels)
        delta[:, 0] = pixels[:, 0]
        delta[:, 1:] = pixels[:, 1:] - pixels[:, :-1]
        predicted = np.repeat(delta[..., None], 3, axis=2)
        strip = _close_lzw(imagecodecs.lzw_encode(predicted.tobytes()))
        strip_offsets.append(8 + len(payload))
        strip_counts.append(len(strip))
        payload.extend(strip)
    if len(payload) % 2:
        payload.append(0)

    ifd_offset = 8 + len(payload)
    # Sixteen entries, a uint16 count and a uint32 next-IFD pointer.
    extra_offset = ifd_offset + 2 + 16 * 12 + 4
    entries, extra = [], bytearray()

    def add_tag(code, kind, values):
        raw = struct.pack('<' + {3: 'H', 4: 'I', 5: 'I'}[kind] * len(values), *values)
        length = len(values) // 2 if kind == 5 else len(values)
        if len(raw) <= 4:
            value = raw.ljust(4, b'\0')
        else:
            value = struct.pack('<I', extra_offset + len(extra))
            extra.extend(raw)
        entries.append(struct.pack('<HHI', code, kind, length) + value)

    add_tag(254, 4, [0])
    add_tag(256, 3, [width])
    add_tag(257, 3, [height])
    add_tag(258, 3, [8, 8, 8])
    add_tag(259, 3, [5])
    add_tag(262, 3, [2])
    add_tag(273, 4, strip_offsets)
    add_tag(274, 3, [1])
    add_tag(277, 3, [3])
    add_tag(278, 3, [rows_per_strip])
    add_tag(279, 4, strip_counts)
    add_tag(282, 5, [96 << 24, 1 << 24])
    add_tag(283, 5, [96 << 24, 1 << 24])
    add_tag(284, 3, [1])
    add_tag(296, 3, [2])
    add_tag(317, 3, [2])
    return (b'II*\0' + struct.pack('<I', ifd_offset) + bytes(payload) +
            struct.pack('<H', 16) + b''.join(entries) + bytes(4) + bytes(extra))
