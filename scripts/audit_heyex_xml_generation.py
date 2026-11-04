"""Generate complete XML bundles from E2E and compare independent HEYEX exports.

Each manifest entry needs e2e, xml and align_followup_to_reference.
reference_e2e is optional for self-references and required for external
references. IDs, software and visit/UTC context are explicitly
synthetic; paired targets are used only for comparisons and the separate
XML serialization check. Unknown metadata and alignment are never
borrowed from those targets. Full cross-format fidelity remains
unverified.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np

from eyepy.io import HeE2eReader
from eyepy.io import HeXmlExportContext
from eyepy.io import HeXmlReader
from eyepy.io import HeXmlWriter
from eyepy.io.he._xml_export import serialize_heyex_xml


def asset(reader, image):
    name = image.findtext('ImageData/ExamURL').replace('\\', '/').rsplit('/', 1)[-1]
    return reader.path.parent / name


def comparison(a, b):
    return dict(byte_identical=a == b, predicted_bytes=len(a), reference_bytes=len(b),
                predicted_sha256=hashlib.sha256(a).hexdigest(),
                reference_sha256=hashlib.sha256(b).hexdigest())


def audit(pair):
    target = HeXmlReader(pair['xml'])
    # Isolate serialization from conversion: these target metadata are NOT
    # used by the source-derived writer below.
    result = dict(full_cross_format_writer_verified=False,
                  reference_document_reserialization=comparison(
                      serialize_heyex_xml(target.xml_root), target._source.read()),
                  context='synthetic database IDs, software, visit date and UTC bias; not verified')
    with warnings.catch_warnings(), ExitStack() as sources, tempfile.TemporaryDirectory(prefix='heyex-xml-generation-') as temporary:
        warnings.simplefilter('ignore', UserWarning)
        reader = sources.enter_context(HeE2eReader(pair['e2e']))
        series, = (s for s in reader.series if s.get_bscan_meta())
        reference = series
        if series.get_reference_key() is not None:
            if 'reference_e2e' in pair:
                reference_reader = sources.enter_context(HeE2eReader(pair['reference_e2e']))
                reference = series.resolve_reference(reference_reader.series)
            else:
                reference = series.resolve_reference()
        volume = series.get_volume(reference=reference if series.get_localizer_transform() is not None else None)
        context = HeXmlExportContext(
            pid=1, study_id=2, series_id=3,
            software_name='eyepy validation', software_version='synthetic',
            align_followup_to_reference=pair['align_followup_to_reference'],
            reference_id=3 if reference is series else 2,
            reference_exam_date=reference.get_meta()['exam_time'],
            visit_date='2000-01-01', utc_bias_minutes=0)
        output = Path(temporary) / 'bundle'
        try:
            HeXmlWriter(volume, export_context=context, include_segmentation=False).write(output)
        except NotImplementedError:
            result['generation'] = 'unsupported external follow-up alignment'
            assert not output.exists()
            return result
        generated = HeXmlReader(output)
        restored = generated.volume
        assert restored.shape == volume.shape
        rows = []
        target_images = {int(image.findtext('ImageNumber')): image
                         for image in target.xml_root.findall('.//Series/Image')}
        generated_images = {int(image.findtext('ImageNumber')): image
                            for image in generated.xml_root.findall('.//Series/Image')}
        assert generated_images.keys() == target_images.keys()
        source_clocks_equal, quality_equal, averages_equal, scale_fields_equal, endpoint_fields_equal = 0, 0, 0, 0, 0
        for number, image in generated_images.items():
            reference_image = target_images[number]
            kind = image.findtext('ImageType/Type')
            assert kind == reference_image.findtext('ImageType/Type')
            expected_pixels = volume.localizer.data if number == 0 else volume[number - 1].data
            restored_pixels = restored.localizer.data if number == 0 else restored[number - 1].data
            np.testing.assert_array_equal(restored_pixels, expected_pixels)
            rows.append(dict(image_number=number, kind=kind,
                             file=comparison(asset(generated, image).read_bytes(),
                                             asset(target, reference_image).read_bytes())))
            if number:
                clock_paths = ['AcquisitionTime/Time/' + key for key in ('Hour', 'Minute', 'Second')]
                source_clocks_equal += int(all(float(image.findtext(path)) == float(reference_image.findtext(path)) for path in clock_paths))
                for field, counts in [('ImageQuality', 'quality'), ('NumAve', 'averages')]:
                    path = 'OphthalmicAcquisitionContext/' + field
                    equal = image.findtext(path) == reference_image.findtext(path)
                    if counts == 'quality':
                        quality_equal += int(equal)
                    else:
                        averages_equal += int(equal)
                for point in ('Start', 'End'):
                    for axis in ('X', 'Y'):
                        path = 'OphthalmicAcquisitionContext/' + point + '/Coord/' + axis
                        endpoint_fields_equal += int(image.findtext(path) == reference_image.findtext(path))
            for axis in ('X', 'Y'):
                path = 'OphthalmicAcquisitionContext/Scale' + axis
                scale_fields_equal += int(image.findtext(path) == reference_image.findtext(path))
        result.update(
            generation='complete generated bundle; complete HEYEX metadata not yet reproduced',
            reimport_passed=True,
            complete_xml=comparison(generated._source.read(), target._source.read()),
            files=len(rows) + 1,
            entire_oct_tiffs_equal=sum(r['file']['byte_identical'] for r in rows if r['kind'] == 'OCT'),
            entire_localizer_tiffs_equal=sum(r['file']['byte_identical'] for r in rows if r['kind'] == 'LOCALIZER'),
            source_time_of_day_equal=source_clocks_equal,
            utc_bias_verified=False, source_quality_equal=quality_equal,
            source_averaging_equal=averages_equal,
            source_pixel_scale_fields_equal=scale_fields_equal,
            source_endpoint_fields_equal=endpoint_fields_equal,
            source_birthdate_equal=generated.parsed_values['dob'] == target.parsed_values['dob'],
            source_focus_equal=generated.parsed_values['scan_focus'] == target.parsed_values['scan_focus'],
            source_laterality_equal=generated.parsed_values['scan_position'] == target.parsed_values['scan_position'],
            source_patient_fields_equal={key: generated.patient.get(key) == target.patient.get(key)
                                         for key in ('firstname', 'lastname', 'dob', 'patient_id', 'sex')},
            images_detail=rows)
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
        if not isinstance(pair.get('align_followup_to_reference'), bool):
            parser.error('Every pair needs an explicit align_followup_to_reference boolean')
        row = dict(example=alias, **audit(pair))
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'images_detail'}), flush=True)
        gc.collect()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(examples=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
