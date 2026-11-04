"""Exercise complete generated VOL files against paired HEYEX exports.

Images and acquisition metadata come exclusively from E2E. Receiving-
database IDs and visit timestamps use explicitly synthetic context, not
target values. The report distinguishes matching source-derived fields
from those deliberately unverified context fields. Calibration, layers
and localizers remain measured differences; a passing OCT comparison
does not establish full writer fidelity.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import tempfile
import warnings

from audit_heyex_paired_exports import aggregate
from audit_heyex_paired_exports import difference
import numpy as np

from eyepy.io import HeE2eReader
from eyepy.io import HeVolExportContext
from eyepy.io import HeVolReader
from eyepy.io import HeVolWriter


def audit(pair):
    with tempfile.TemporaryDirectory(prefix='heyex-vol-generation-') as temporary:
        output = Path(temporary) / 'generated.vol'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with ExitStack() as sources:
                reader = sources.enter_context(HeE2eReader(pair['e2e']))
                series, = (s for s in reader.series if s.get_bscan_meta())
                reference_series = None
                if 'reference_e2e' in pair:
                    reference_reader = sources.enter_context(HeE2eReader(pair['reference_e2e']))
                    reference_series = series.resolve_reference(reference_reader.series)
                volume = series.get_volume(reference=reference_series)
                # These are test-context values, never read from the target.
                context = HeVolExportContext(pid=1, vid=1, id='1', reference_id='1',
                                             visit_date='2000-01-01T00:00:00',
                                             exam_time='2000-01-01T00:00:00'
                                             if volume.meta['exam_time'] is None else None,
                                             scan_pattern=3)
                HeVolWriter(volume, export_context=context).write(output)
        generated_reader = HeVolReader(output)
        reference_reader = HeVolReader(pair['vol'])
        generated, reference = generated_reader.parsed_file, reference_reader.parsed_file
        oct_rows, layer_rows, quality_rows, matrix_rows, positions = [], [], [], [], []
        entire_oct_equal = 0
        for a, b in zip(generated.bscans, reference.bscans, strict=True):
            oct_rows.append(difference(a.data, b.data, float_bits=True))
            entire_oct_equal += int(np.array_equal(a.data.view('<u4'), b.data.view('<u4')))
            layer_rows.append(difference(a.layer_segmentations, b.layer_segmentations, float_bits=True))
            quality_rows.append(difference([a.quality], [b.quality], float_bits=True))
            matrix_rows.append(difference(a.iv_transformation, b.iv_transformation, float_bits=True))
            positions.append(difference([a.start_x, a.start_y, a.end_x, a.end_y],
                                        [b.start_x, b.start_y, b.end_x, b.end_y]))
        raw_generated = generated_reader._source.read(0, 2048)
        raw_reference = reference_reader._source.read(0, 2048)
        return dict(
            context='synthetic receiving-database IDs and visit timestamp; not verified',
            full_cross_format_writer_verified=False,
            reference_calibration_supplied='reference_e2e' in pair,
            source_exam_time_available=volume.meta['exam_time'] is not None,
            header=dict(version_equal=generated.version == reference.version,
                        bscan_versions_equal=all(a.version == b.version for a, b in zip(generated.bscans, reference.bscans)),
                        bscan_header_size_equal=generated.bscan_hdr_size == reference.bscan_hdr_size,
                        segmentation_counts_equal=all(a.num_seg == b.num_seg for a, b in zip(generated.bscans, reference.bscans)),
                        source_birthdate_bits_equal=raw_generated[164:172] == raw_reference[164:172],
                        source_exam_time_bits_equal=raw_generated[88:96] == raw_reference[88:96]
                        if volume.meta['exam_time'] is not None else None,
                        progression_id_equal=generated.prog_id == reference.prog_id),
            entire_oct_images_equal=entire_oct_equal,
            oct=aggregate(oct_rows), layers=aggregate(layer_rows),
            quality=aggregate(quality_rows), registration=aggregate(matrix_rows),
            endpoints=aggregate(positions),
            localizer=difference(generated.localizer, reference.localizer),
            calibration=difference([generated.scale_x, generated.scale_y, generated.distance,
                                    generated.scale_x_slo, generated.scale_y_slo],
                                   [reference.scale_x, reference.scale_y, reference.distance,
                                    reference.scale_x_slo, reference.scale_y_slo]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; refusing to overwrite')
    examples = []
    for alias, pair in json.loads(args.manifest.read_text()).items():
        if 'vol' not in pair:
            continue
        row = dict(example=alias, **audit(pair))
        examples.append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(examples=examples), indent=2) + '\n')


if __name__ == '__main__':
    main()
