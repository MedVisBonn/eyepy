"""Verify TIFF bytes against paired exports, distinguishing two independent
checks.

The codec check re-encodes reference pixels to isolate TIFF
serialization. The conversion check renders exclusively from E2E source
data, then encodes and compares complete TIFF bytes. Neither check
establishes complete XML metadata or cross-format writer fidelity. Uses
the same private manifest as audit_heyex_paired_exports.py. Reports omit
source paths and patient metadata.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import imageio.v3 as iio
import numpy as np

from eyepy.io import HeE2eReader
from eyepy.io.he._xml_tiff import encode_heyex_tiff


def grayscale(pixels):
    if pixels.ndim == 2 and pixels.dtype == np.uint8:
        return pixels
    if pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.dtype == np.uint8:
        if np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 0], pixels[..., 2]):
            return pixels[..., 0]
    raise ValueError('Reference image is outside the grayscale TIFF profile')


def comparison(predicted, reference):
    return dict(byte_identical=predicted == reference,
                predicted_bytes=len(predicted), reference_bytes=len(reference),
                predicted_sha256=hashlib.sha256(predicted).hexdigest(),
                reference_sha256=hashlib.sha256(reference).hexdigest())


def verify(pair):
    path = Path(pair['xml'])
    images = sorted(ET.parse(path).getroot().findall('.//Series/Image'),
                    key=lambda image: int(image.findtext('ImageNumber')))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with HeE2eReader(pair['e2e']) as reader:
            series, = (s for s in reader.series if s.get_bscan_meta())
            volume = series.get_volume()
            bscan_index = 0
            for image in images:
                name = image.findtext('ImageData/ExamURL').replace('\\', '/').rsplit('/', 1)[-1]
                reference = (path.parent / name).read_bytes()
                pixels = grayscale(iio.imread(reference, extension='.tif'))
                codec = encode_heyex_tiff(pixels)
                kind = image.findtext('ImageType/Type')
                if kind == 'LOCALIZER':
                    rendered = volume.localizer.data
                elif kind == 'OCT':
                    rendered = volume[bscan_index].data
                    bscan_index += 1
                else:
                    raise ValueError('Unexpected paired image type')
                converted = encode_heyex_tiff(rendered)
                rows.append(dict(kind=kind, image_number=int(image.findtext('ImageNumber')),
                                 reference_pixel_reencoding=comparison(codec, reference),
                                 source_e2e_conversion=comparison(converted, reference)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; refusing to overwrite')
    examples = []
    for alias, pair in json.loads(args.manifest.read_text()).items():
        if 'xml' not in pair:
            continue
        rows = verify(pair)
        summary = dict(example=alias, images=len(rows),
                       reference_pixel_reencoding_exact=sum(row['reference_pixel_reencoding']['byte_identical'] for row in rows),
                       source_e2e_conversion_exact=sum(row['source_e2e_conversion']['byte_identical'] for row in rows))
        examples.append(dict(**summary, images_detail=rows))
        print(json.dumps(summary), flush=True)
        gc.collect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(scope='TIFF codec and source-only image conversion',
                                          full_cross_format_writer_verified=False,
                                          examples=examples), indent=2) + '\n')


if __name__ == '__main__':
    main()
