"""Field-preserving native VOL serialization, including opaque/trailing data."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def _header_fields(format, values, stop):
    offset = 0
    for field in format.subcons:
        if field.name == stop:
            break
        size = field.sizeof(**values)
        yield offset, field, size
        offset += size


def capture_layout(reader):
    from eyepy.io.he.vol_reader import bscan_format
    from eyepy.io.he.vol_reader import vol_format

    parsed = reader.parsed_file
    root = {field.name: deepcopy(parsed[field.name])
            for _, field, _ in _header_fields(vol_format, parsed, 'localizer')}
    headers = []
    for record in parsed.bscans:
        headers.append({field.name: deepcopy(record[field.name]) for _, field, _ in
                        _header_fields(bscan_format, record, 'layer_segmentations')})
    return root, headers


def _equal(first, second):
    try:
        return np.array_equal(first, second, equal_nan=True)
    except TypeError:  # NumPy's equal_nan is not defined for string fields.
        return np.array_equal(first, second)


def _header_patches(format, current, original, stop, start=0):
    for offset, field, size in _header_fields(format, original, stop):
        value = current[field.name]
        if not _equal(value, original[field.name]):
            encoded = field.build(value, **current)
            if len(encoded) != size:
                raise ValueError('Native VOL field edits must retain their size')
            yield start + offset, encoded


def patches(reader):
    from eyepy.io.he.vol_reader import bscan_format
    from eyepy.io.he.vol_reader import vol_format

    current = reader.parsed_file
    root, headers = reader._native_layout
    for name in ('size_x', 'size_y', 'n_bscans', 'size_x_slo', 'size_y_slo',
                 'bscan_hdr_size'):
        if current[name] != root[name]:
            raise ValueError('Native VOL rewriting cannot change the image layout')
    if len(current.bscans) != len(headers):
        raise ValueError('Native VOL B-scan count differs from the source')
    yield from _header_patches(vol_format, current, root, 'localizer')
    localizer = np.asarray(current.localizer)
    if localizer.dtype != np.uint8 or localizer.shape != (root['size_y_slo'], root['size_x_slo']):
        raise ValueError('Native VOL localizer must retain its uint8 shape')
    yield 2048, localizer.tobytes()
    position = 2048 + localizer.size
    for record, original in zip(current.bscans, headers):
        for name in ('bscan_hdr_size', 'seg_offset', 'num_seg'):
            if record[name] != original[name]:
                raise ValueError('Native VOL rewriting cannot change the B-scan layout')
        yield from _header_patches(bscan_format, record, original,
                                   'layer_segmentations', position)
        segments = np.asarray(record.layer_segmentations)
        if segments.shape != (original['num_seg'], root['size_x']):
            raise ValueError('Native VOL layer shape differs from the source')
        yield position + original['seg_offset'], segments.astype('<f4', copy=False).tobytes()
        image = np.asarray(record.data)
        if image.shape != (root['size_y'], root['size_x']):
            raise ValueError('Native VOL B-scan shape differs from the source')
        position += original['bscan_hdr_size']
        encoded = image.astype('<f4', copy=False).tobytes()
        yield position, encoded
        position += len(encoded)
