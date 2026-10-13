# Heidelberg Engineering E2E format

E2E is a container format with patients, studies, series, and slices. A file may
contain several OCT acquisitions, empty series, localizers, and annotations.
Use the reader to inspect it before selecting an acquisition.

```python
from eyepy.io import HeE2eReader

print(HeE2eReader("scan.e2e"))
with HeE2eReader("scan.e2e") as reader:
    volume = reader.volume      # first supported OCT EyeVolume
    volumes = reader.volumes    # list of supported OCT EyeVolumes
    series = next(s for s in reader.series if s.n_bscans)
    metadata = series.get_heyex_metadata(bscan_index=0)
    sources = series.get_heyex_metadata_sources(bscan_index=0)
    inputs = series.get_calibration_metadata()
```

[`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader] must stay open while accessing
its series and payloads. Returned EyeVolumes retain their data after closing it.
`reader.series[0]` can be empty. `reader.volume` is a single EyeVolume, while
`reader.volumes` is a list; the reader does not accept `single=True`.

## Calibration

Default E2E imports use pixel units. Focus, corneal radius, preset, angular
geometry, progression/reference keys, and registration are decoded, but the
exact vendor optical equation remains unverified. Metric lateral fields in the
HEYEX helper are unavailable unless an approximate model is explicitly selected.

See [calibration and verified fields](he_e2e_calibration.md) for metric imports,
external reference lookup, binary offsets, migration notes, evidence, and limits.
An unresolved recorded reference never silently uses a follow-up's current focus.
Intensity registration and resampling are separate from coordinate calibration.

## Hierarchy and other payloads

`reader.file_hierarchy` exposes the hierarchy. Convenience attributes return
lists of `reader.patients`, `reader.studies`, and `reader.series`. Each series has
`slices` and `folders`; each folder provides its bounded `get_bytes()` payload and
parsed `.data`. Inline directory values have no container payload.

```python
with HeE2eReader("scan.e2e") as reader:
    print(reader.inspect(recursive=True))
```

The generated [file-structure reference](he_e2e_structures/he_e2e_structure_doc.md)
describes the hierarchy and individual container layouts. The volume metadata
bundle at `volume.meta["e2e_metadata"]` contains HEYEX-style fields, per-B-scan OCT
metadata, their sources, and calibration provenance. Per-B-scan ART, quality,
and acquisition time require the `bscan_index` argument.

## Acknowledgements

The reader and investigation build on
[OCT-Converter](https://github.com/marksgraham/OCT-Converter),
[LibE2E](https://github.com/neurodial/LibE2E),
[uocte](https://bitbucket.org/uocte/uocte/wiki/Home),
[RETIMAT](https://github.com/drombas/retimat), and the optical source cited in the
calibration documentation.
