"""Verify full-byte native E2E/VOL/XML rewrites against a private manifest.

Manifest: {"e2e": {"alias": "/input.e2e"}, "vol": {...}, "xml": {...}}.
Reports contain aliases, counts and hashes, never input paths or patient
fields. This verifies native rewriting only; cross-format equivalence is
a separate gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from eyepy.io import HeE2eReader
from eyepy.io import HeE2eWriter
from eyepy.io import HeVolReader
from eyepy.io import HeVolWriter
from eyepy.io import HeXmlReader
from eyepy.io import HeXmlWriter


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(kind, path, scratch):
    readers = {'e2e': HeE2eReader, 'vol': HeVolReader, 'xml': HeXmlReader}
    writers = {'e2e': HeE2eWriter, 'vol': HeVolWriter, 'xml': HeXmlWriter}
    reader = readers[kind](path)
    writer = writers[kind](reader)
    sources = ({reader.path.name: writer.source.path,
                **{name: source.path for name, source in writer.assets.items()}}
               if kind == 'xml' else {'export': Path(path)})
    before = {name: digest(source) for name, source in sources.items()}
    with tempfile.TemporaryDirectory(prefix='heyex-native-', dir=scratch) as name:
        target = Path(name) / ('bundle' if kind == 'xml' else f'export.{kind}')
        writer.write(target)
        outputs = ({name: target / name for name in sources} if kind == 'xml'
                   else {'export': target})
        equal = all(digest(outputs[name]) == checksum for name, checksum in before.items())
        unchanged = all(digest(sources[name]) == checksum for name, checksum in before.items())
        assert equal, 'Native output differs from its source'
        assert unchanged, 'A native source changed during validation'
        loaded = readers[kind](target)
        if kind == 'e2e':
            assert len(reader.file_hierarchy.all_folders) == len(loaded.file_hierarchy.all_folders)
        elif kind == 'vol':
            assert len(reader.parsed_file.bscans) == len(loaded.parsed_file.bscans)
        else:
            assert reader.parsed_values == loaded.parsed_values
        row = dict(files=len(sources), bytes=sum(p.stat().st_size for p in sources.values()),
                   file_hashes=before, byte_identical=True, sources_unchanged=True,
                   reimport_passed=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; refusing to overwrite')
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for kind, entries in manifest.items():
        for alias, path in entries.items():
            row = dict(format=kind, example=alias, **verify(kind, path, args.scratch))
            # XML asset names may contain identifying fields; publish only an
            # ordered list of checksums rather than those native filenames.
            row['file_hashes'] = sorted(row['file_hashes'].values())
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('format', 'example', 'files', 'byte_identical')}), flush=True)
            gc.collect()
    report = dict(scope='full native file bytes and all XML referenced assets',
                  cross_format_verified=False, examples=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
