from pathlib import Path
import struct

import numpy as np
import pytest

from eyepy.io import HeE2eReader
from eyepy.io import HeE2eWriter
from eyepy.io import HeVolReader
from eyepy.io import HeVolWriter
from eyepy.io import HeXmlReader
from eyepy.io import HeXmlWriter
from eyepy.io.he.e2e_format import Bscan
from eyepy.io.he.e2e_format import ContainerHeader
from eyepy.io.he.e2e_format import containerheader_format
from eyepy.io.he.e2e_format import FolderHeader
from eyepy.io.he.e2e_format import folderheader_format
from eyepy.io.he.e2e_format import Header
from eyepy.io.he.e2e_format import header_format
from eyepy.io.he.e2e_format import TypesEnum
from eyepy.io.he.e2e_format import Version
from eyepy.io.he.e2e_format import version_format

VOL = Path('tests/data/filetypes/heyex_vol/test_volume.vol')
XML = Path('tests/data/filetypes/heyex_xml/test_volume')


def e2e_fixture(path):
    # A native directory with an inline value, opaque metadata and an OCT
    # image. Two distinct stored zero words deliberately decode identically.
    words = np.array([1, 1023, 0xf801, 0xffff], dtype='<u2')
    image = struct.pack('<5I', 8, 35652097, 4, 1, 4) + words.tobytes()
    payloads = [None, b'opaque\xffmetadata', image]
    types = [42001, 42002, 1073741824]
    main = dict(magic2='MDbMDir', version=100, unknown0=[0] * 10,
                num_entries=3, current=88, prev=200,
                unknown1=(0x56789abc + 3 + 88 + 200) & 0xffffffff)
    chunk = dict(main, magic2='MDbDir', current=88, prev=0,
                 unknown1=(0x3456789a + 3 + 88) & 0xffffffff)
    prefix = version_format.build(Version(name='CMDb', version=100, unknown0=[0] * 10))
    prefix += header_format.build(Header(**main)) + header_format.build(Header(**chunk))
    directory, containers = [], []
    position = 88 + 52 + len(payloads) * 44
    for index, (kind, payload) in enumerate(zip(types, payloads)):
        folder_pos = 88 + 52 + index * 44
        start, size = (1, 0) if payload is None else (position, len(payload))
        checksum = (0x87654321 + folder_pos + start + size + 1 + 2 + 3 + 1 + kind) & 0xffffffff
        entry = dict(pos=folder_pos, start=start, size=size, unknown0=0,
                     patient_id=1, study_id=2, series_id=3, slice_id=0, ind=1,
                     unknown1=0, type=TypesEnum(kind), unknown3=checksum)
        directory.append(folderheader_format.build(FolderHeader(**entry)))
        if payload is not None:
            header = dict(magic3='MDbData', unknown0=(0x12435687 + sum(payload)) & 0xffffffff,
                          header_pos=folder_pos, pos=start, size=size, unknown1=0,
                          patient_id=1, study_id=2, series_id=3, slice_id=0,
                          ind=1, unknown2=0, type=TypesEnum(kind), unknown3=123456)
            containers.append(containerheader_format.build(ContainerHeader(**header)) + payload)
            position += 60 + len(payload)
    data = prefix + b''.join(directory) + b''.join(containers)
    path.write_bytes(data)
    return data


def test_native_e2e_roundtrip_preserves_inline_opaque_and_zero_words(tmp_path):
    source = tmp_path / 'source.e2e'
    expected = e2e_fixture(source)
    with HeE2eReader(source) as reader:
        image = next(f for f in reader.file_hierarchy.all_folders if int(f.type) == 1073741824)
        np.testing.assert_array_equal(image.data.data[0, :2], [0, 0])
        writer = HeE2eWriter(reader)
        # Re-encoding unchanged decoded pixels must retain noncanonical zero words.
        writer.replace_bscan(image, image.data.data)
    assert writer.bytes == expected
    output = tmp_path / 'copy.e2e'
    writer.write(output)
    assert output.read_bytes() == expected


def test_native_e2e_edits_reencode_only_changed_pixels_and_checksum(tmp_path):
    source = tmp_path / 'source.e2e'
    before = e2e_fixture(source)
    with HeE2eReader(source) as reader:
        image = next(f for f in reader.file_hierarchy.all_folders if int(f.type) == 1073741824)
        values = image.data.data.copy()
        values[0, -1] = .5
        writer = HeE2eWriter(reader)
        writer.replace_bscan(image, values)
    output = tmp_path / 'edited.e2e'
    writer.write(output)
    after = output.read_bytes()
    start = image.start + 60 + 20
    np.testing.assert_array_equal(np.frombuffer(after, '<u2', offset=start, count=4),
                                  [1, 1023, 0xf801, 0xf800])
    changed = {index for index, (a, b) in enumerate(zip(before, after)) if a != b}
    assert changed <= set(range(image.start + 12, image.start + 16)) | {start + 6, start + 7}
    with HeE2eReader(output) as loaded:
        folder = next(f for f in loaded.file_hierarchy.all_folders if int(f.type) == 1073741824)
        assert folder.data.data[0, -1] == .5
        payload = folder.get_bytes()
        assert folder.header.unknown0 == (0x12435687 + sum(payload)) & 0xffffffff
    with pytest.raises(ValueError, match='exactly representable'):
        writer.replace_bscan(image, np.full((1, 4), .10001))
    with pytest.raises(ValueError, match='payload size'):
        writer.replace_payload(image, b'invalid')


def test_native_vol_preserves_headers_precision_spare_bytes_and_trailer(tmp_path):
    raw = bytearray(VOL.read_bytes())
    raw[1024:1034] = b'OPAQUEVOL!'
    # Preserve date bits finer than Python datetime's microsecond resolution.
    stamp = struct.unpack_from('<Q', raw, 88)[0]
    struct.pack_into('<Q', raw, 88, stamp + 1)
    original = HeVolReader(VOL)
    position = 2048 + original.parsed_file.localizer.size
    raw[position + 100:position + 110] = b'UNKNOWNHDR'
    pixel = position + original.parsed_file.bscans[0].bscan_hdr_size
    struct.pack_into('<I', raw, pixel, 0x7fc12345)
    raw.extend(b'opaque thickness-grid trailer\x00\xff')
    source = tmp_path / 'source.vol'
    source.write_bytes(raw)
    reader = HeVolReader(source)
    writer = HeVolWriter(reader)
    assert writer.bytes == raw
    output = tmp_path / 'copy.vol'
    writer.write(output)
    assert output.read_bytes() == raw
    # Native writing exposes editable decoded fields without rebuilding unknowns.
    reader.parsed_file.scale_x *= 2
    reader.parsed_file.bscans[0].data = reader.parsed_file.bscans[0].data.copy()
    reader.parsed_file.bscans[0].data[0, 1] = .125
    edited = tmp_path / 'edited.vol'
    HeVolWriter(reader).write(edited)
    changed = {i for i, (a, b) in enumerate(zip(raw, edited.read_bytes())) if a != b}
    assert changed <= set(range(24, 32)) | set(range(pixel + 4, pixel + 8))
    assert HeVolReader(edited).parsed_file.bscans[0].data[0, 1] == .125


def test_native_vol_honors_nondefault_segmentation_offset(tmp_path):
    reader = HeVolReader(VOL)
    raw = VOL.read_bytes()
    first = 2048 + reader.parsed_file.localizer.size
    original_header = reader.parsed_file.bscans[0].bscan_hdr_size
    modified = bytearray(raw[:first + 256] + b'OPAQUE!!' + raw[first + 256:])
    struct.pack_into('<I', modified, first + 12, original_header + 8)
    struct.pack_into('<I', modified, first + 52, 264)
    source = tmp_path / 'offset.vol'
    source.write_bytes(modified)
    native = HeVolReader(source)
    np.testing.assert_array_equal(native.parsed_file.bscans[0].layer_segmentations,
                                  reader.parsed_file.bscans[0].layer_segmentations)
    assert HeVolWriter(native).bytes == modified


def test_native_xml_bundle_preserves_document_and_encoded_images(tmp_path):
    reader = HeXmlReader(XML)
    writer = HeXmlWriter(reader)
    output = tmp_path / 'bundle'
    writer.write(output)
    assert (output / reader.path.name).read_bytes() == reader.path.read_bytes()
    for name, source in writer.assets.items():
        assert (output / name).read_bytes() == source.path.read_bytes()
    loaded = HeXmlReader(output)
    np.testing.assert_array_equal(loaded.volume.raw_data, reader.volume.raw_data)
    # Reader trees must not be shared through a mutable global parser cache.
    reader.body.find('./SWVersion/Version').text = 'edited'
    assert HeXmlReader(XML).parsed_values['version'] != 'edited'
    edited = tmp_path / 'edited'
    writer.write(edited)
    assert HeXmlReader(edited).parsed_values['version'] == 'edited'


def test_native_writers_protect_sources_destinations_and_changed_inputs(tmp_path):
    source = tmp_path / 'source.e2e'
    e2e_fixture(source)
    writer = HeE2eWriter(HeE2eReader(source))
    with pytest.raises(ValueError, match='overwrite.*source'):
        writer.write(source, overwrite=True)
    occupied = tmp_path / 'occupied.e2e'
    occupied.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        writer.write(occupied)
    assert occupied.read_bytes() == b'existing'
    source.write_bytes(source.read_bytes() + b'changed')
    output = tmp_path / 'changed.e2e'
    with pytest.raises(ValueError, match='source changed'):
        writer.write(output)
    assert not output.exists()


def test_native_vol_failed_layout_edit_does_not_publish_output(tmp_path):
    reader = HeVolReader(VOL)
    reader.parsed_file.size_x += 1
    output = tmp_path / 'invalid.vol'
    with pytest.raises(ValueError, match='image layout'):
        HeVolWriter(reader).write(output)
    assert not output.exists()


def test_xml_comments_and_processing_instructions_survive_metadata_edits(tmp_path):
    bundle = tmp_path / 'source'
    HeXmlWriter(HeXmlReader(XML)).write(bundle)
    path = next(bundle.glob('*.xml'))
    path.write_bytes(path.read_bytes().replace(
        b'<HEDX>', b'<HEDX><!--opaque--><?processor opaque?>', 1))
    reader = HeXmlReader(bundle)
    reader.body.find('./SWVersion/Version').text = 'edited'
    output = tmp_path / 'edited'
    HeXmlWriter(reader).write(output)
    written = (output / path.name).read_bytes()
    assert b'<!--opaque-->' in written
    assert b'<?processor opaque?>' in written
    assert HeXmlReader(output).parsed_values['version'] == 'edited'
