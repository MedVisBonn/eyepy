"""Source-backed serialization helpers for native Heidelberg exports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import io
import os
from pathlib import Path
import tempfile
from typing import BinaryIO


@dataclass(frozen=True)
class SourceFile:
    """A source file whose identity must remain stable during serialization.

    Keeping the original backing file avoids duplicating large OCT
    exports in memory. Writers preserve opaque bytes while explicitly
    replacing decoded fields. A changed backing file is rejected, never
    silently copied.
    """

    path: Path
    fingerprint: tuple[int, int, int, int]

    @classmethod
    def capture(cls, path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        return cls(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))

    def verify(self):
        stat = self.path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != self.fingerprint:
            raise ValueError('The native export source changed after it was read')

    @property
    def size(self):
        return self.fingerprint[2]

    def read(self, offset=0, size=None):
        self.verify()
        if size is None:
            size = self.size - offset
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError('Native export read is outside the source file')
        with self.path.open('rb') as source:
            source.seek(offset)
            data = source.read(size)
        self.verify()
        if len(data) != size:
            raise EOFError('Truncated native export source')
        return data

    def rewrite(self, output: BinaryIO, patches: Iterable[tuple[int, bytes]] = ()):
        """Write source bytes plus ordered, fixed-size field replacements."""
        self.verify()
        with self.path.open('rb') as source:
            position = 0
            for offset, data in patches:
                if offset < position or offset + len(data) > self.size:
                    raise ValueError('Native export patches overlap or exceed the source')
                self._copy(source, output, offset - position)
                output.write(data)
                position = offset + len(data)
                source.seek(position)
            self._copy(source, output, self.size - position)
        self.verify()

    @staticmethod
    def _copy(source, output, remaining):
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise EOFError('Truncated native export source')
            output.write(block)
            remaining -= len(block)

    def build(self, patches=()):
        output = io.BytesIO()
        self.rewrite(output, patches)
        return output.getvalue()


def write_file(path, producer, *, sources=(), overwrite=False):
    """Publish a complete export atomically, protecting source files."""
    path = Path(path)
    for source in sources:
        if path.resolve() == source.path or (
                path.exists() and path.samefile(source.path)):
            raise ValueError('Refusing to overwrite the native export source')
    if path.exists() and not overwrite:
        raise FileExistsError('Export destination already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            producer(output)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
