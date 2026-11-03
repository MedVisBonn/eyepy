"""Native E2E serialization retaining undecoded records and encoded OCT words."""

from __future__ import annotations

import struct

import numpy as np

from eyepy.io.he._native import write_file
from eyepy.io.he.e2e_format import Bscan
from eyepy.io.he.e2e_reader import E2EFolder
from eyepy.io.he.e2e_reader import HeE2eReader


class HeE2eWriter:
    """Rewrite an indexed native E2E export without losing opaque records.

    Unchanged files retain every byte, including unknown payloads,
    directory padding, checksums and noncanonical OCT zero words.
    Explicit record edits currently require unchanged payload sizes;
    payload checksums are updated. Constructing a complete E2E database
    from an EyeVolume is not supported.
    """

    def __init__(self, reader: HeE2eReader):
        if not isinstance(reader, HeE2eReader):
            raise TypeError('Native E2E writing requires a HeE2eReader')
        self.reader = reader
        self.source = reader._source
        self._replacements = {}

    def replace_payload(self, folder: E2EFolder, data: bytes):
        """Replace one bounded payload and recompute its native checksum."""
        if folder.reader is not self.reader or not any(
                folder is item for item in self.reader.file_hierarchy.all_folders):
            raise ValueError('Folder does not belong to the source reader')
        if folder.start < 88:
            raise ValueError('Inline directory entries have no container payload')
        data = bytes(data)
        if len(data) != folder.size:
            raise ValueError('Native E2E replacement must retain the payload size')
        original = self.source.read(folder.start, 60 + folder.size)
        checksum = struct.unpack_from('<I', original, 12)[0]
        if checksum != (0x12435687 + sum(original[60:])) & 0xffffffff:
            raise ValueError('Unsupported native E2E payload checksum')
        self._replacements[folder.start] = data

    def replace_bscan(self, folder: E2EFolder, data: np.ndarray):
        """Encode changed raw OCT samples, preserving unchanged source words.

        Input is in the original E2E intensity domain and geometry.
        Samples must be exactly representable by the native unsigned
        float encoding.
        """
        original = self._replacements.get(folder.start)
        if original is None:
            original = self.source.read(folder.start + 60, folder.size)
        if int(folder.type) != 1073741824 or len(original) < 20:
            raise ValueError('Expected an E2E OCT image record')
        _, kind, count, height, width = struct.unpack_from('<5I', original)
        if kind != 35652097 or count != height * width or len(original) != 20 + count * 2:
            raise ValueError('Unsupported E2E OCT image layout')
        data = np.asarray(data)
        if data.shape != (height, width):
            raise ValueError('Replacement B-scan shape differs from the source')
        words = np.frombuffer(original, '<u2', offset=20).reshape(height, width).copy()
        changed = data != Bscan.LUT[words]
        try:
            words[changed] = [Bscan.inv_LUT[value] for value in data[changed]]
        except (KeyError, OverflowError) as exc:
            raise ValueError('Raw samples are not exactly representable in E2E') from exc
        self.replace_payload(folder, original[:20] + words.tobytes())

    def _patches(self):
        for offset, payload in sorted(self._replacements.items()):
            # Container offset 12 is the additive payload checksum, not the
            # directory checksum. Sizes and hierarchy keys remain unchanged.
            checksum = (0x12435687 + sum(payload)) & 0xffffffff
            yield offset + 12, struct.pack('<I', checksum)
            yield offset + 60, payload

    @property
    def bytes(self):
        return self.source.build(self._patches())

    def write(self, path, *, overwrite=False):
        write_file(path, lambda output: self.source.rewrite(output, self._patches()),
                   sources=[self.source], overwrite=overwrite)
