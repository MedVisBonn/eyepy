"""Compare controlled HEYEX XML exports with follow-up alignment on/off.

Manifest entries map reference_aligned, reference_unaligned,
followup_aligned and followup_unaligned to XML paths. Optional
reference_old/followup_old entries compare earlier exports as well. Only
aliases, image numbers, hashes and differences enter the report. No
alignment algorithm is inferred from the comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from audit_heyex_paired_exports import aggregate
from audit_heyex_paired_exports import difference
import imageio.v3 as iio
from verify_heyex_tiff_encoding import grayscale

from eyepy.io import HeXmlReader
from eyepy.io.he._native import SourceFile


def capture(path):
    reader = HeXmlReader(path)
    images = {}
    for image in reader.xml_root.findall('.//Series/Image'):
        number = int(image.findtext('ImageNumber'))
        if number in images:
            raise ValueError('Duplicate XML image number')
        name = image.findtext('ImageData/ExamURL').replace('\\', '/').rsplit('/', 1)[-1]
        if name in {'', '.', '..', reader.path.name}:
            raise ValueError('Invalid XML asset filename')
        source = SourceFile.capture(reader.path.parent / name)
        images[number] = dict(source=source,
                              kind=image.findtext('ImageType/Type'),
                              sha256=hashlib.sha256(source.read()).hexdigest(),
                              clock=image.find('AcquisitionTime'))
    # Replacing only image URLs retains every other XML byte, including
    # whitespace, tag ordering, identifiers and acquisition metadata.
    normalized_xml = re.sub(rb'(?<=<ExamURL>)[^<]*(?=</ExamURL>)',
                            b'IMAGE_URL', reader._source.read())
    return dict(reader=reader, images=images, normalized_xml=normalized_xml)


def compare(first, second):
    from xml.etree.ElementTree import tostring

    if first['images'].keys() != second['images'].keys():
        raise ValueError('The controlled exports have different image numbers')
    rows, counts, clock_matches = [], Counter(), 0
    for number, a in first['images'].items():
        b = second['images'][number]
        if a['kind'] != b['kind']:
            raise ValueError('The controlled exports have different image types')
        a_clock = tostring(a['clock']) if a['clock'] is not None else None
        b_clock = tostring(b['clock']) if b['clock'] is not None else None
        clock_matches += int(a_clock == b_clock)
        identical = a['sha256'] == b['sha256']
        counts[a['kind'] + '_files'] += 1
        counts[a['kind'] + '_byte_identical'] += int(identical)
        row = dict(image_number=number, kind=a['kind'], byte_identical=identical)
        if not identical:
            # Decode only changed images; immutable source handles verify the
            # exact bytes read before and after each image comparison.
            left = grayscale(iio.imread(a['source'].read(), extension='.tif'))
            right = grayscale(iio.imread(b['source'].read(), extension='.tif'))
            row['pixels'] = difference(left, right)
        rows.append(row)
    pixel_summary = {}
    for kind in ('OCT', 'LOCALIZER'):
        changed = [r['pixels'] for r in rows if r['kind'] == kind and 'pixels' in r]
        pixel_summary[kind] = dict(
            changed_encoding_comparisons=aggregate(changed),
            decoded_pixels_equal=sum(r['byte_identical'] or
                                     (r.get('pixels', {}).get('shape_equal', False) and
                                      r['pixels']['unequal_valid'] == 0 and
                                      r['pixels']['missing_mask_differences'] == 0)
                                     for r in rows if r['kind'] == kind))
    return dict(images=len(rows), counts=dict(counts), acquisition_clocks_equal=clock_matches,
                xml_identical_except_image_urls=first['normalized_xml'] == second['normalized_xml'],
                pixels=pixel_summary, images_detail=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; refusing to overwrite')
    manifest = json.loads(args.manifest.read_text())
    required = ('reference_aligned', 'reference_unaligned',
                'followup_aligned', 'followup_unaligned')
    if any(alias not in manifest for alias in required):
        parser.error('Manifest must contain the four controlled export aliases')
    exports = {alias: capture(path) for alias, path in manifest.items()}
    pairs = [('reference_aligned', 'reference_unaligned'),
             ('followup_aligned', 'followup_unaligned')]
    for kind in ('reference', 'followup'):
        if kind + '_old' in exports:
            pairs.extend((kind + '_' + mode, kind + '_old')
                         for mode in ('aligned', 'unaligned'))
    rows = []
    for a, b in pairs:
        row = dict(first=a, second=b, **compare(exports[a], exports[b]))
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'images_detail'}), flush=True)
    for export in exports.values():
        export['reader']._source.verify()
        for image in export['images'].values():
            image['source'].verify()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(
        scope='controlled export differences; no inferred alignment implementation',
        full_cross_format_writer_verified=False, sources_unchanged=True,
        comparisons=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
