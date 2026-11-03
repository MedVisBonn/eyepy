"""Audit conversion components against paired exports, without borrowing targets.

Manifest: {"alias": {"e2e": "/source.e2e", "vol": "/reference.vol",
"xml": "/reference.xml"}}. VOL/XML are optional per pair. Reference
paths, names and patient fields are excluded from the report. This is a
diagnostic audit, not a claim that output writers are fully verified.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import warnings

import imageio.v3 as iio
import numpy as np

from eyepy.io import HeE2eReader
from eyepy.io import HeVolReader
from eyepy.io import HeXmlReader
from eyepy.io.he.vol_reader import SEG_MAPPING


def difference(prediction, reference, *, float_bits=False):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        return dict(shape_equal=False, predicted_shape=list(prediction.shape),
                    reference_shape=list(reference.shape))
    missing_a = ~np.isfinite(prediction) | (prediction >= 3e38)
    missing_b = ~np.isfinite(reference) | (reference >= 3e38)
    valid = ~missing_a & ~missing_b
    a, b = prediction[valid], reference[valid]
    unequal = (a.astype('<f4').view('<u4') != b.astype('<f4').view('<u4')
               if float_bits else a != b)
    return dict(shape_equal=True, samples=int(reference.size),
                missing_mask_differences=int(np.count_nonzero(missing_a != missing_b)),
                unequal_valid=int(np.count_nonzero(unequal)),
                maximum_absolute_error=float(np.max(abs(a.astype(float) - b))) if a.size else 0.)


def aggregate(rows):
    if not rows:
        return dict(comparisons=0)
    return dict(comparisons=len(rows), shape_mismatches=sum(not r['shape_equal'] for r in rows),
                samples=sum(r.get('samples', 0) for r in rows),
                unequal_valid=sum(r.get('unequal_valid', 0) for r in rows),
                missing_mask_differences=sum(r.get('missing_mask_differences', 0) for r in rows),
                maximum_absolute_error=max(r.get('maximum_absolute_error', 0) for r in rows))


def audit(pair):
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with HeE2eReader(pair['e2e']) as reader:
            series, = (s for s in reader.series if s.get_bscan_meta())
            volume = series.get_volume()
            if 'vol' in pair:
                reference = HeVolReader(pair['vol']).parsed_file
                oct_rows, layers, positions, qualities, ui_qualities = [], [], [], [], []
                for index, bscan in enumerate(reference.bscans):
                    oct_rows.append(difference(volume[index].linear_data, bscan.data, float_bits=True))
                    for name, layer in volume.layers.items():
                        if SEG_MAPPING[name] < len(bscan.layer_segmentations):
                            layers.append(difference(layer.data[index], bscan.layer_segmentations[SEG_MAPPING[name]], float_bits=True))
                    meta = volume[index].meta
                    positions.append(difference([*meta['start_pos'], *meta['end_pos']],
                                                [bscan.start_x, bscan.start_y, bscan.end_x, bscan.end_y]))
                    qualities.append(difference([meta['quality']], [bscan.quality]))
                    ui_qualities.append(difference([meta['quality_ui']], [bscan.quality]))
                result['e2e_to_vol_components'] = dict(
                    oct=aggregate(oct_rows), layers=aggregate(layers), endpoints=aggregate(positions),
                    quality_legacy=aggregate(qualities), quality_ui=aggregate(ui_qualities),
                    localizer=difference(volume.localizer.data, reference.localizer),
                    calibration=difference([volume.scale_x, volume.scale_y, volume.scale_z,
                                            volume.localizer.scale_x, volume.localizer.scale_y],
                                           [reference.scale_x, reference.scale_y, reference.distance,
                                            reference.scale_x_slo, reference.scale_y_slo]))
                del reference
            if 'xml' in pair:
                xml = HeXmlReader(pair['xml'])
                images = sorted(xml.body.findall(".//ImageType[Type='OCT'].."),
                                key=lambda element: int(element.findtext('ImageNumber')))
                oct_rows = []
                for index, image in enumerate(images):
                    name = image.findtext('ImageData/ExamURL').replace('\\', '/').rsplit('/', 1)[-1]
                    target = iio.imread(xml.path.parent / name)
                    predicted = volume[index].data
                    if target.ndim == 3:
                        predicted = np.repeat(predicted[..., None], target.shape[-1], axis=-1)
                    oct_rows.append(difference(predicted, target))
                result['e2e_to_xml_components'] = dict(
                    oct=aggregate(oct_rows),
                    localizer=difference(volume.localizer.data, xml.localizer.raw_data))
            result['full_cross_format_writer_verified'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; refusing to overwrite')
    rows = []
    for alias, pair in json.loads(args.manifest.read_text()).items():
        row = dict(example=alias, **audit(pair))
        rows.append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(scope='conversion components; complete writers not verified',
                                          examples=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
