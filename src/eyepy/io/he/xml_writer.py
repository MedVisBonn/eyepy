"""Native Heidelberg XML export bundles, retaining original image encodings."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ElementTree

from eyepy.io.he._native import SourceFile
from eyepy.io.he._native import write_file
from eyepy.io.he.xml_reader import HeXmlReader


class HeXmlWriter:
    """Rewrite a native XML document and every referenced image asset.

    An unchanged XML tree retains the exact original document bytes.
    Assets retain their original bytes, including TIFF tags/compression
    and JPEG encoding. XML edits are serialized as XML; semantic changes
    cannot retain the original document's byte identity. The destination
    is a new directory.
    """

    def __init__(self, reader: HeXmlReader):
        if not isinstance(reader, HeXmlReader):
            raise TypeError('Native XML writing requires a HeXmlReader')
        self.reader = reader
        self.source = reader._source
        self.assets = {}
        for element in reader.xml_root.findall('.//ImageData/ExamURL'):
            if element.text is None:
                raise ValueError('Empty XML image reference')
            name = element.text.replace('\\', '/').rsplit('/', 1)[-1]
            if not name or name in {'.', '..', reader.path.name}:
                raise ValueError('Invalid XML image filename')
            if name not in self.assets:
                self.assets[name] = SourceFile.capture(reader.path.parent / name)
        self._references = self._image_references()

    def _image_references(self):
        return tuple(element.text for element in
                     self.reader.xml_root.findall('.//ImageData/ExamURL'))

    @property
    def bytes(self):
        current = ElementTree.tostring(self.reader.xml_root, encoding='utf-8')
        if current == self.reader._xml_snapshot:
            return self.source.read()
        # Keep explicit XML declarations when the source supplied one.
        declaration = self.source.read(0, min(5, self.source.size)) == b'<?xml'
        return ElementTree.tostring(self.reader.xml_root, encoding='utf-8',
                                    xml_declaration=declaration)

    def write(self, path):
        """Atomically create a new directory containing the complete bundle."""
        path = Path(path)
        if path.exists():
            raise FileExistsError('XML export destination already exists')
        if self._image_references() != self._references:
            raise ValueError('Native XML image references cannot be changed without replacement assets')
        self.source.verify()
        for source in self.assets.values():
            source.verify()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.', dir=path.parent))
        try:
            xml = self.bytes
            write_file(temporary / self.reader.path.name,
                       lambda output: output.write(xml))
            for name, source in self.assets.items():
                write_file(temporary / name, source.rewrite)
            self.source.verify()
            for source in self.assets.values():
                source.verify()
            if path.exists():
                raise FileExistsError('XML export destination already exists')
            temporary.rename(path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
