# E2E calibration, references, and verified fields

Decoded acquisition inputs are separate from optical constants. The exact
vendor optical equation remains unverified. Default E2E imports retain
`scale_unit='px'` and unit volume scales; lateral metric fields in the HEYEX
metadata helper are `None` unless a model is explicitly selected.

## Explicit approximate calibration

```python
from eyepy.io import HeE2eReader
from eyepy.io.he.e2e_calibration import EMPIRICAL_OPTICAL_MODEL

with HeE2eReader("scan.e2e") as reader:
    series = next(s for s in reader.series if s.n_bscans)
    inputs = series.get_calibration_metadata()
    volume = series.get_volume(
        optical_model=EMPIRICAL_OPTICAL_MODEL,
        localizer_angle_deg=30,
    )
    provenance = volume.meta["e2e_metadata"]["findings"]["calibration"]
    assert provenance["status"] == "approximate"
```

This path derives lateral scale from the full stored image width, axial scale
from the stored mm/pixel value, and raster spacing from reference-frame centers.
Localizer and B-scan positions use the same mm/degree coefficient and top-left
origin. It rejects nonuniform sampling and circular/radial patterns. It does not
resample OCT intensities or decode the B-scan intensity-registration pipeline.
For a reference-linked metric volume, the localizer image is taken from the
resolved reference so that its pixels and the reference-frame endpoints agree.
The current acquisition's OCT data and annotations are retained. Default imports
keep the acquisition's own localizer. This path assumes both acquisitions use
the declared localizer field; differing fields need separate validation.

The default 30-degree localizer field is an assumption retained from the original
reader, not a decoded universal field of view. Supply the actual field when known.
Calling `get_volume()` again without a model returns the pixel-unit volume.

[`E2EOpticalModel`][eyepy.io.he.e2e_calibration.E2EOpticalModel] evaluates

```text
q_mm_per_degree = 1 / (a / corneal_radius_mm + b + c * focus_diopters)
scale_x_mm_per_pixel = q_mm_per_degree * angular_span_deg / full_width_px
```

Two approximate models are provided in `eyepy.io.he.e2e_calibration`:

| Model | Provenance and limits |
| --- | --- |
| `RESEARCH_OPTICAL_MODEL` | Public [FoveaAnalysis source](https://github.com/halirutan/FoveaAnalysis/blob/5b3cb4a3e5021043f20cd41a87655b1b996f082a/FoveaAnalysis/Modeling.m#L89-L125): `1.02302 / (17.21/R + 1.247 + focus/17.455)`. Matches supplied rounded displays after reference selection but fails finer XML-coordinate constraints. |
| `EMPIRICAL_OPTICAL_MODEL` | Four-point fit: `a=16.828362441469007`, `b=1.2182834015811586`, `c=0.05588751262262184`. Consistent with the coordinates used to fit it; no independent coordinate-level holdout. |

The empirical points cover radii 6.2, 7.7, and 9.0 mm and focuses -1.51, -0.49,
+0.82, and +0.90 D. Other ranges and firmware need validation. Neither model's
constants were decoded from E2E. A validated custom model can be supplied as a
named `E2EOpticalModel`. This is not individual anatomical magnification correction.

## Follow-ups and exact reference keys

```text
current key:   (Type10005 UUID at +0, examination timestamp at +16)
reference key: (Type10025 UUID at +8, reference timestamp at +24)
```

UUID bytes use Windows byte order. Timestamp keys are uint64 and must not pass
through floating-point or rounded datetime conversion. Lookup is verified against
explicit reference IDs in one paired XML export. Type 6 ordering and Type 10025's
high flag bit do not establish reference direction.

References in the same file can resolve automatically. For external references:

```python
from eyepy.io import HeE2eReader
from eyepy.io.he.e2e_calibration import EMPIRICAL_OPTICAL_MODEL

with HeE2eReader("reference.e2e") as baseline_reader, \
        HeE2eReader("followup.e2e") as followup_reader:
    followup = next(s for s in followup_reader.series if s.n_bscans)
    baseline = followup_reader.resolve_reference(
        followup, additional_series=baseline_reader.series,
    )
    if baseline is None:
        raise ValueError("The recorded reference is unavailable")
    volume = followup.get_volume(
        optical_model=EMPIRICAL_OPTICAL_MODEL,
        reference_series=baseline,
        localizer_angle_deg=30,
    )
```

Both readers must remain open while their series are accessed. An unresolved
recorded reference leaves metric calibration unavailable; the follow-up's current
focus is not substituted. A mismatched supplied reference or ambiguous lookup
raises an error. Absence of progression records is reported as `not_recorded`;
that alone does not prove independence.

For the observed version-3 Type 10025 affine `[L | t]`,

```text
registered_angle = L @ reference_angle + t * localizer_angle / localizer_width
reference_angle = inverse(L) @ (registered_angle - t * localizer_angle / localizer_width)
```

The linear part acts on centered angles; translation is in centered localizer
pixels. Do not add a top-left offset before the inverse. Use
`series.get_reference_angles(localizer_angle_deg=30)` without an optical model.
The supplied pair maps all 242 follow-up endpoints within approximately 9.3e-7
degrees. Unsupported layouts and singular inverses are explicit errors.

## Binary fields and migration

Offsets are relative to the container payload, excluding the 60-byte header in
the investigated files. Numeric fields are little-endian unless stated.

| Type | Offset | Encoding | Interpretation |
| --- | --- | --- | --- |
| 10004 | 4, 8 | uint32 | Full height/width; agrees with image headers. |
| 10004 | 12, 16, 20, 24 | float32 | Start/end x/y in centered degrees. |
| 10004 | 36 | float32 | Axial mm/pixel. |
| 10004 | 56, 60 | uint32 | Margin/filled-coverage candidates, not full sampling width. |
| 10004 | 68 | uint32 | B-scan image index used for ordering. |
| 10004 | **136** | **float64** | Raw focus; empirical conversion `raw * float32(1.043) / 1000`. |
| 10004 | **144** | **float64** | Preset code: -2500 Short, 0 Medium, 2500 Long, 5000 Extra Long. |
| 10004 | 156 | float32 | Per-B-scan HEYEX quality. |
| 39 | **457** | **float64** | Corneal radius in mm, confirmed by controlled 9.0/6.2 mm settings. |
| 7 | 0, **17** | ASCII L/R, **float64** | Study eye and corneal radius in mm. |
| 10005 | 0, 16 | UUID bytes_le, uint64 | Progression identity and current timestamp key. |
| 10025 | 0, 4 | uint32 | Observed version 3 and size/flags; low 31 bits equal payload length, high bit unknown. |
| 10025 | 8, 24, 32 | UUID bytes_le, uint64, six float32 | Progression identity, reference timestamp, affine. |

The former float32 interpretation at +140 reads only half the double at +136.
Its nonlinear focus fit is superseded. `focus_candidate_raw` now returns the
corrected double. The fixed `289.6 um/degree` and filled-width scale calculation
are removed. Full width, rather than width minus one, reproduces the reported
scale convention for the supplied horizontal reference grid.

Type 7's old names were incorrect: +17 was labeled cylinder and +1 corneal
curvature. The parser now calls +17 `c_curve_mm`, keeps +1/+9 unknown, and
preserves the undecoded remainder. Acquisition Type 39 radius takes precedence;
the matching study eye is a fallback, with conflicts recorded. Both eyes share
settings in the controlled exports, so eye-specific conflict resolution remains
untested. Preset values are not measured axial lengths; their raw units and
general influence on scaling remain unresolved. Unknown preset codes stay unlabeled.

## Other records and file structure

* **Type 6:** terminated UTF-16LE label, uint16 count, then 18-byte entries.
  `folder.data.membership()` exposes patient/study/series membership and preserves
  undecoded entry fields. Reference direction is not inferred.
* **Type 2:** 16-byte prefix and JPEG thumbnail in the supplied files.
* **Types 1073751824/1073751825/1073751826:** auxiliary image arrays with standard
  image headers; particular geometric roles remain unknown.
* **Type 10020:** opaque record. Opening one acquisition added a 20-byte payload
  without changing any existing payload or the resulting EyeVolume.
* **Type 10012:** possible image-registration fields; intensity resampling remains
  undecoded.

In the supplied opening pair, the new Type 10020 container adds 80 bytes including
its header, using one empty directory slot. Two existing records cross chunk
boundaries, changing offsets, back-pointers, links, and checksums. Complete byte
accounting confirms all 1412 existing payloads and six inline values are identical.
This does not establish Type 10020 semantics or universal HEYEX behavior.

Small `FolderHeader.start` values below the data region are inline values, not
file offsets. `E2EFolder.is_inline` exposes this distinction; payload access for
an inline entry raises an error. Parsing is bounded to the declared container,
so trailing unknown bytes cannot consume neighboring records. Truncation and
directory/container size disagreement are explicit errors.

## Provenance and validation boundary

HEYEX-style metadata and sources remain available through
`get_heyex_metadata()` and `get_heyex_metadata_sources()`. Per-B-scan ART count,
quality, and acquisition time require `bscan_index`. These helpers accept the
same calibration options. The volume's `e2e_metadata` bundle records decoded
inputs, keys, affine, effective reference inputs, assumptions, and selected model.

The investigation covers eleven exports, nine acquisitions, 1017 unique B-scans,
and five XML exports matching 533 B-scans. XML ScaleX is rounded to 0.0001 mm/pixel;
coordinates retain 0.001 mm precision. TIFF display exports are not pixel-identical
to direct E2E rendering, so matching metadata does not establish identical image
processing. Synthetic regression fixtures contain no patient data.

The public optical model fails finer coordinate constraints by roughly 4–11 ppm
at the first 7.7 mm calibration and 27–49/69–88 ppm at the 9.0/6.2 mm controls,
even with free origins. One global multiplier cannot correct all points. The
reciprocal fit remains empirical.

Whole-file searches over 1.66 GB found no binary float32/float64 literal optical
constants. Metadata interval searches found no fitted coefficients, tested
reciprocal/scaled equivalents, or XML-constrained effective scales. Known focus
and radius fields served as positive controls. Apparent hits were image-byte
coincidences or nonconstant fixed-point interpretations. Software-defined
constants are plausible; arbitrary encodings, compression, and other
parameterizations remain possible.

Other unresolved mappings include objective/target settings, A-scan rate, EDI/EVI,
timezone interpretation, and distinct camera/FPGA firmware assignment. No vendor
re-import compatibility is claimed.
