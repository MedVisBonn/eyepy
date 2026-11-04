# Heidelberg export fidelity

This work is separate from PR #84. The branch builds on its intensity-domain
and registration changes; it does not modify that PR's branch.

## Required completion gates

1. Readers and writers for native E2E, VOL, and XML export bundles.
2. Unmodified native read/write is byte-identical, including opaque metadata,
   reserved bytes, encoded image words, TIFF/JPEG encoding, and XML formatting.
3. Cross-format output matches paired HEYEX exports from the same acquisition.
   Verification must cover images, localizers, layer coordinates and missing
   values, calibration, endpoints, metadata, and serialization. A matching OCT
   image alone is insufficient.
4. Reference exports are verification inputs, not a source of missing output
   pixels or metadata during conversion.

These gates are independent. Native byte preservation does not establish
cross-format fidelity or compatibility with HEYEX's import implementation.

## Native rewriting

```python
from eyepy.io import HeE2eReader, HeE2eWriter
from eyepy.io import HeVolReader, HeVolWriter
from eyepy.io import HeXmlReader, HeXmlWriter

with HeE2eReader('source.e2e') as reader:
    HeE2eWriter(reader).write('copy.e2e')

reader = HeVolReader('source.vol')
HeVolWriter(reader).write('copy.vol')

reader = HeXmlReader('source.xml')
HeXmlWriter(reader).write('new_export_directory')
```

The native writers retain their backing sources rather than copying large
exports into additional in-memory snapshots. Keep source files unchanged until
writing finishes. A changed source is rejected. Output files are published
only after serialization succeeds, and existing destinations are rejected by
default. Native writers cannot overwrite their source files. Binary writers
accept `overwrite=True` for another existing destination. The XML writer
creates a new directory containing the document and every referenced image.

For VOL, decoded header fields and pixel/layer arrays in `parsed_file` are
editable; the writer patches those fields while preserving unrelated bytes.
Native layout changes are currently rejected. Unlike a complete native rewrite,
`HeVolWriter(volume)` constructs a VOL from an `EyeVolume` and cannot recover
opaque records that the volume does not contain.

For E2E, `replace_payload(folder, payload)` supports explicit fixed-size
payload edits and updates the checksum. `replace_bscan(folder, raw_samples)`
encodes changed samples in the original E2E geometry and intensity domain.
Unchanged samples retain their original unsigned-float words, including
noncanonical zero representations. Unrepresentable samples and layout-changing
edits are rejected. Constructing an E2E database from an `EyeVolume` remains
unfinished.

For XML, unchanged documents retain their original bytes and unchanged images
retain all their original encoded bytes. Metadata edits to `reader.xml_root`
are serialized as XML, preserving comments and processing instructions inside
the root. `replace_image(filename, display)` explicitly replaces an existing
TIFF with a 2D uint8 grayscale image of the dimensions recorded in the XML.
It uses the observed HEYEX RGB TIFF encoding profile and requires `eyepy[tiff]`.
The caller supplies the rendered image; the writer does not infer its intensity
curve or registration. Adding or changing image references is still rejected.
Complete XML generation from a different format remains unfinished.

```python
reader = HeXmlReader('source.xml')
writer = HeXmlWriter(reader)
writer.replace_image(reader.bscan_meta[0]['scan_name'], rendered_bscan)
writer.write('edited_export_directory')
```

## Verification

The private-fixture native audit exercises 15 E2E files, 4 VOL files, and
5 XML bundles (543 XML/image files), for 562 files in total. Every rewritten
file matches its original SHA-256, original sources remain unchanged, and
rewritten exports re-import successfully. Synthetic tests additionally cover
opaque header/trailer bytes, sub-microsecond timestamps, NaN payloads,
noncanonical E2E zeros, changed pixels/checksums, nondefault segmentation
offsets, and source/destination protection.

### TIFF serialization

The TIFF encoder reconstructs the observed HEYEX layout from rendered pixels:
little-endian RGB grayscale, LZW with horizontal prediction, 8 KiB nominal
strips, 16 explicitly ordered IFD entries, 96 dpi rationals, and zero alignment
padding. It also reproduces HEYEX's terminal LZW code width when the final data
code lies at a dictionary-growth boundary. It does not copy compressed data
or TIFF metadata from the paired target.

Two checks deliberately isolate encoding from rendering:

* Re-encoding the decoded pixels of all 538 reference TIFFs reconstructs every
  complete TIFF byte-for-byte. This proves serialization for those fixtures;
  it does not prove E2E conversion.
* Rendering from E2E and encoding those pixels reconstructs 416 complete TIFFs
  byte-for-byte across four ordinary paired exports (412 OCT and 4 localizers).
  The follow-up's 122 TIFFs remain different and are included as failures.

Both checks pass with imagecodecs 2025.3.30 and 2026.8.16. The implementation
uses the minimum supported LZW API rather than depending on newer TIFF-wrapper
arguments. Run them with:

```bash
PYTHONPATH=src python scripts/verify_heyex_tiff_encoding.py manifest.json report.json
```

The report records image hashes and explicitly leaves full cross-format writer
verification false. XML metadata generation, calibration and follow-up geometry
are separate requirements.

### Controlled follow-up alignment exports

Four additional XML bundles exported with Spectralis Viewing Module 7.0.7.0
isolate the XML export option **Align follow-up to reference**. They contain
the baseline and follow-up acquisitions, each exported with the option on and
off. Acquisition clocks agree with the earlier exports for all 122 images per
bundle. The comparisons establish the following complete-TIFF results:

| Acquisition / XML export option | E2E-derived OCT TIFFs exact | E2E-derived localizer TIFFs exact |
| --- | ---: | ---: |
| Baseline, on | 121 / 121 | 1 / 1 |
| Baseline, off | 121 / 121 | 1 / 1 |
| Follow-up, off | 121 / 121 | 0 / 1 |
| Follow-up, on | 0 / 121 | 0 / 1 |

The baseline's 122 TIFFs are byte-identical between the two settings and to
the earlier baseline export. The follow-up's earlier 121 OCT TIFFs are
byte-identical to the new export with alignment **on**. Every follow-up OCT
image changes when the option is switched off, while its localizer stays
byte-identical. That localizer's decoded pixels equal the native VOL localizer;
our E2E localizer resampling still differs at 6843 pixels, independently of this
export option.

The complete XML documents also match byte-for-byte after replacing only the
`ExamURL` contents. This holds between on/off controls and between the earlier
exports and the new on controls. Consequently these XML documents do not record
which of the two image-processing settings produced their OCT assets. A writer
must receive this export setting explicitly; it cannot infer it from those
metadata fields alone.

The existing Type10012 normalization/registration pipeline reproduces the
follow-up export with this option **off**. Turning the option off does not mean
disabling Type10012 registration. Turning it on adds another OCT processing
step, whose algorithm remains unresolved. Likewise, `get_volume(reference=...)`
selects metric output calibration; it does not implement this extra XML export
alignment.

All 488 new reference TIFFs are re-encoded exactly from their own decoded pixels,
including the aligned follow-up images. Thus their encoding profile is already
supported; the aligned image-generation step is the remaining difference.
Native rewriting of all four bundles also passes: 492 XML/asset files match
their originals and sources remain unchanged. Unreferenced database/cache files
in an export directory are outside the XML bundle.

Run `scripts/audit_heyex_alignment_exports.py manifest.json report.json` with
the aliases `reference_aligned`, `reference_unaligned`, `followup_aligned` and
`followup_unaligned` mapped to XML paths. Optional `reference_old` and
`followup_old` entries compare earlier exports. It reports per-image byte and
pixel comparisons and checks source fingerprints, without recording patient
fields or input paths. The separate native and TIFF verifiers exercise the
writers and source-derived rendering; the alignment audit itself compares
controls and does not supply an inferred alignment implementation.

Run the audit with a private manifest mapping format names to alias/path pairs:

```bash
PYTHONPATH=src python scripts/verify_heyex_native_roundtrip.py manifest.json report.json
```

The report contains aliases, sizes and hashes. It explicitly records that it
has not verified cross-format conversion. Input paths and patient metadata
are excluded from the report.

## Remaining cross-format work

PR #84's reference checks establish exact OCT-linear reconstruction for 460
paired VOL B-scans and exact display for 412 ordinary E2E/TIFF comparisons.
They do not establish the complete export gate. In particular, nonidentity
localizer interpolation, follow-up XML display geometry, vendor layer
interpolation/endpoints, exact calibration, and complete export metadata still
need independent verification and implementation. Display-only XML lacks the
original linear samples; reverse conversions must identify what source data
or provenance is required rather than silently inventing it.

`scripts/audit_heyex_paired_exports.py` compares the current conversion
components against private paired exports and explicitly marks complete
cross-format writers as unverified. Its first audit confirmed the OCT pixel
matches and identified the remaining image, layer, geometry and calibration
differences. It also established across all 460 VOL B-scans that Type10004's
`quality_ui` matches exported VOL quality exactly; the historical `quality`
field does not. The VOL writer now uses that decoded UI quality when present.

### Follow-up acquisition references

Some E2E series identify their acquisition with a Type10005 key and identify
their reference acquisition in Type10025. The key combines a progression UUID
with the original 64-bit Windows timestamp. Matching on a Python `datetime`
would discard the final timestamp digit and can conflate distinct acquisitions.

```python
with HeE2eReader('baseline.e2e') as baseline_reader, HeE2eReader('followup.e2e') as followup_reader:
    baseline = baseline_reader.series[0]
    followup = followup_reader.series[0]
    reference = followup.resolve_reference([baseline, followup])
    volume = followup.get_volume(reference=reference)
```

Keep both readers open while accessing their series. `resolve_reference()`
returns the series itself when there is no reference key or the key explicitly
refers to the same acquisition. An external reference must resolve to exactly
one supplied series. Missing candidates, ambiguous matches and malformed keys
raise `ValueError`; filenames and scan dates are never used to guess a match.
`get_acquisition_key()` and `get_reference_key()` expose the immutable keys for
applications that manage their own acquisition index.

Passing the resolved acquisition to `get_volume(reference=...)` uses its
localizer scales for the registered output grid. `get_localizer(registered=True,
reference=...)` provides the same operation for the localizer alone. Pixels
still come from the follow-up image. Its own calibration remains in
`localizer.meta['raw_calibration']`; the usual `scale_x`, `scale_y`, `scale_unit`
and `field_size` describe the registered output. B-scan `raw_start_pos` and
`raw_end_pos` remain in the acquisition's metric frame. Both calibration and
reference provenance survive `.eye` serialization.
Materializing registration with `EyeEnface.transform()` removes the old raw
calibration because the returned object's raw pixels are now the transformed
output.

The input pixel position is computed with the acquisition's scale, transformed
to output pixels, then converted to physical coordinates with the reference's
scale. Replacing both scales with the reference scale would incorrectly change
the input pixel positions. No target VOL/XML scale is used. Both acquisitions
still use the documented approximate optical model: this corrects the reference
frame, not the unresolved vendor metric equation.

Explicit references must match the native key and eye. Different-sized grids,
different units and references that themselves need an external reference are
rejected until their composition is implemented. Without `reference=`, the
reader retains its prior acquisition-based metric approximation and warns when
the registered localizer identifies another acquisition. Raw views do not need
the reference and remain readable independently.

In the private paired fixtures the follow-up resolves to the supplied baseline,
while four other series have no reference records. The generated VOL audit
accepts an optional `reference_e2e` source path per manifest pair; the resolver
validates it before use. The paired VOL remains a verification input only.

Using the supplied baseline reduces the follow-up's maximum endpoint difference
from about 0.2895 mm to 0.00006284 mm. The remaining error is consistent with
the still approximate metric calibration; endpoint bytes are not exact. All
460 generated OCT arrays remain bit-identical to the paired VOLs, and the
follow-up's 6843 localizer pixel differences remain visible in the report.

### Patient metadata

Type9's birth date is a float64 OLE Automation day count at offset 93, between
the 11-byte title and the sex byte at offset 101. The reader now exposes that
complete value as `birthdate`, and `date_of_birth` returns its ISO calendar date.
The historical layout read only its upper four bytes as an integer. Decoded
dates match all five paired XML exports and all four paired VOL exports.
Unknown trailing patient-record bytes remain available as `rest`.

### Generated VOL files and database context

Importing an E2E into another HEYEX database can assign new patient, visit and
scan IDs. The supplied pairs were exported after such an import. Matching those
IDs requires the receiving database's context; acquisition identity does not
determine its local counters or visit timestamp.

```python
from eyepy.io import HeE2eReader, HeVolExportContext, HeVolWriter

context = HeVolExportContext(
    pid=101, vid=202, id='303', reference_id='302',
    visit_date='2026-01-01T09:00:00', scan_pattern=3,
)
with HeE2eReader('source.e2e') as reader:
    volume = reader.series[0].get_volume()
    HeVolWriter(volume, export_context=context).write('generated.vol')
```

The identifiers and timestamp above are examples, not acquisition-derived
defaults. Optional context fields override source metadata when supplied.
`patient_id` is the receiving database's user-defined patient identifier;
`pid`, `vid`, `id` and `reference_id` are its internal identifiers. E2E and VOL
scan-pattern enums differ, so E2E conversion requires an explicit VOL
`scan_pattern` when no VOL-native value is available. A self-reference maps to
the newly assigned scan ID; external references require `reference_id` from the
receiving database. Missing examination timestamps and visit timestamps must
be supplied explicitly rather than substituted with a B-scan timestamp or date.

Generated files use `HSF-OCT-103` and `HSF-BS-103` signatures, the available
segmentation slots, and `0x7f7fffff` for missing heights. B-scan matrices are
retained as metadata alongside already registered pixels. Layer export reads
the volume annotations directly; it does not depend on which lazy B-scan layer
views the caller has accessed. Later layer slots 17 and 18 are supported.
Writing streams one generated B-scan at a time. A newly generated header uses
the observed `80 * width` allocation, enlarged when required to fit its vectors;
known source header sizes and segmentation offsets are retained when available.

VOL imports now carry known identifiers, scan pattern, registration metadata
and exact timestamp/day-count scalars through `EyeVolume` and `.eye` saving.
Those scalar encodings are retained only while the corresponding metadata is
unchanged; explicit edits or context overrides take precedence. This preserves
sub-microsecond examination time bits that a `datetime` cannot represent.
The E2E and VOL Windows-time adapters use integer arithmetic for representable
microseconds, avoiding the floating-point drift of the previous adapters.
For exact receiving-database scalars, context accepts `exam_time_ticks` (100 ns
units since 1601-01-01) and `visit_date_days` (float64 OLE Automation days).
Supply the raw scalar or its ISO/datetime counterpart, not both. Explicit raw
context values retain every encoded bit; an explicit datetime replaces any
retained source encoding even when its displayed time is unchanged.
Uninterpreted reserved bytes and trailing grids still require native
`HeVolWriter(reader)` rewriting.

`scripts/audit_heyex_vol_generation.py` writes and reimports complete VOL files
using E2E source data and explicitly synthetic database context. Across all four
paired VOL exports, all 460 entire OCT arrays match in every float32 bit,
including invalid padding. Quality and six registration coefficients also match
for all 460 B-scans; signatures, B-scan header sizes, segmentation counts and
birth-date scalar bytes match every pair. Examination time bytes match the two
E2Es that provide Type10005 timestamps. The other two timestamps and all local
database IDs remain unverified context values. Layer geometry, calibration,
endpoints and the follow-up localizer still differ. The audit reports these
differences and leaves full cross-format writer verification false.
