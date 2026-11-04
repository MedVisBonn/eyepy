"""Native Heidelberg XML export bundles, retaining original image encodings."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ElementTree

import numpy as np

from eyepy.io.he._native import SourceFile
from eyepy.io.he._native import write_file
from eyepy.io.he._xml_export import GeneratedXmlBundle
from eyepy.io.he._xml_export import HeXmlExportContext
from eyepy.io.he.xml_reader import HeXmlReader


class HeXmlWriter:
    """Write a native XML bundle or generate one from an EyeVolume.

    An unchanged XML tree retains the exact original document bytes.
    Unchanged assets retain their original bytes, including TIFF
    tags/compression and JPEG encoding. Explicit TIFF replacements use
    the observed HEYEX encoding profile. XML edits are serialized as
    XML; semantic changes cannot retain the original document's byte
    identity. The destination is a new directory.

    An EyeVolume input requires explicit HeXmlExportContext. Its display
    pixels are encoded as TIFFs and its metadata/layers form a new XML
    document. Exact complete HEYEX metadata and follow-up alignment are
    not yet reproduced; unsupported external alignment is rejected.
    ``include_segmentation=False`` selects an export without layer
    records, as in the supplied paired XML exports.
    """

    def __init__(self, reader, *, export_context: HeXmlExportContext | None = None,
                 include_segmentation: bool = True):
        from eyepy.core.eyevolume import EyeVolume

        self._generated = None
        if isinstance(reader, EyeVolume):
            self._generated = GeneratedXmlBundle(reader, export_context, include_segmentation)
            return
        if not isinstance(reader, HeXmlReader):
            raise TypeError('XML writing requires a HeXmlReader or EyeVolume')
        if export_context is not None or not include_segmentation:
            raise ValueError('Native XML metadata is edited through reader.xml_root, not export context')
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
        self._replacements = {}

    def replace_image(self, name: str, display: np.ndarray):
        """Replace a referenced TIFF with rendered uint8 grayscale pixels.

        ``name`` is the asset's filename, without directories.
        Dimensions must agree with every corresponding XML image entry.
        No display curve, registration or calibration is inferred here.
        """
        from eyepy.io.he._xml_tiff import encode_heyex_tiff

        if self._generated is not None:
            raise ValueError('Edit the source EyeVolume to replace generated images')
        if name not in self.assets:
            raise ValueError('Image is not referenced by the source XML')
        if Path(name).suffix.lower() not in {'.tif', '.tiff'}:
            raise ValueError('Replacement encoding is only available for TIFF images')
        display = np.asarray(display)
        shapes = []
        for image in self.reader.xml_root.findall('.//Image'):
            reference = image.findtext('ImageData/ExamURL')
            if reference is not None and reference.replace('\\', '/').rsplit('/', 1)[-1] == name:
                context = image.find('OphthalmicAcquisitionContext')
                if context is None or context.findtext('Height') is None or context.findtext('Width') is None:
                    raise ValueError('XML image dimensions are required for replacement')
                shapes.append((int(context.findtext('Height')), int(context.findtext('Width'))))
        if not shapes or any(display.shape != shape for shape in shapes):
            raise ValueError('Replacement image dimensions disagree with the XML')
        self._replacements[name] = encode_heyex_tiff(display)

    def _image_references(self):
        return tuple(element.text for element in
                     self.reader.xml_root.findall('.//ImageData/ExamURL'))

    @property
    def bytes(self):
        if self._generated is not None:
            return self._generated.bytes
        current = ElementTree.tostring(self.reader.xml_root, encoding='utf-8')
        if current == self.reader._xml_snapshot:
            return self.source.read()
        # Keep explicit XML declarations when the source supplied one.
        declaration = self.source.read(0, min(5, self.source.size)) == b'<?xml'
        return ElementTree.tostring(self.reader.xml_root, encoding='utf-8',
                                    xml_declaration=declaration)

    def write(self, path):
        """Atomically create a new directory containing the complete bundle."""
        if self._generated is not None:
            return self._generated.write(path)
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
                if name in self._replacements:
                    payload = self._replacements[name]
                    write_file(temporary / name, lambda output: output.write(payload))
                else:
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
