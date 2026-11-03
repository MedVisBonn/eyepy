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
```

Keep both readers open while accessing their series. `resolve_reference()`
returns the series itself when there is no reference key or the key explicitly
refers to the same acquisition. An external reference must resolve to exactly
one supplied series. Missing candidates, ambiguous matches and malformed keys
raise `ValueError`; filenames and scan dates are never used to guess a match.
`get_acquisition_key()` and `get_reference_key()` expose the immutable keys for
applications that manage their own acquisition index.

This resolves provenance only. It does not change calibration or establish an
exact metric conversion. In the private paired fixtures the follow-up resolves
to the supplied baseline, while four other series have no reference records.

### Patient metadata

Type9's birth date is a float64 OLE Automation day count at offset 93, between
the 11-byte title and the sex byte at offset 101. The reader now exposes that
complete value as `birthdate`, and `date_of_birth` returns its ISO calendar date.
The historical layout read only its upper four bytes as an integer. Decoded
dates match all five paired XML exports and all four paired VOL exports.
Unknown trailing patient-record bytes remain available as `rest`.
