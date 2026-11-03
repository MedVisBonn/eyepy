# Heidelberg Engineering VOL Format

While eyepy can read most information from the VOL format, a comprehensive documentation is still missing. If you are interested in contributing to eyepy, documenting a file format is a great way to get started. If you are interested, contact us via GitHub by opening an [issue](https://github.com/MedVisBonn/eyepy/issues).

## Linear intensities and display

`volume.raw_data` retains the imported float32 linear intensities. The default
`volume.data` display applies this curve in float64, then converts to uint8:

```python
gray = clip(floor(30.776116889447362 * log1p(4096 * intensity)
                  + 0.001931325892041094), 0, 255)
```

These are empirical global coefficients, not decoded vendor constants. Five
B-scans from one acquisition determined the coefficients. They reproduced
all pixels in the valid overlap of 218 matched VOL/TIFF B-scans, including
213 withheld images and a separate acquisition. Subsequent E2E reconstruction
reproduced the complete pixel arrays of 460 VOL B-scans and 412 ordinary
XML-associated TIFFs, including padding, using the same display curve.
The exports used default HEYEX brightness and contrast. Other display settings
and software versions remain unverified.

The curve clips finite input to [0, 1]. Float32 maximum (`0x7F7FFFFF`), NaN
and infinities render black. The input array is never modified.
`vol_linear_to_display()` implements this rendering.
`display_to_vol_linear_approx()` chooses an intensity in the middle of a
display bin; it cannot recover intensities lost during conversion to 8-bit
greys. The previous names `from_vol_intensity()` and `to_vol_intensity()`
remain deprecated aliases.

## Intensity encoding and export

`volume.intensity_encoding` describes the samples in `raw_data`, independently
of display settings:

| Encoding | Stored samples | `linear_data` |
| --- | --- | --- |
| `vol_linear` | Imported VOL-linear values | Copy of the stored values |
| `e2e_raw` | Original decoded E2E values | Supported normalization/registration, or raises `ValueError` |
| `display` | Display greys, such as XML-associated TIFFs | Raises `ValueError` |
| `unknown` | Unspecified domain | Raises `ValueError` |

Check `volume.linear_data_available` before requesting linear data. `data`
continues to provide a view for every encoding, including display-only inputs
and raw E2E fallbacks. For arrays supplied directly, declare their encoding
with `EyeVolume(data, intensity_encoding='vol_linear')`, or use `display` for
display greys. The default is `unknown`; dtype, value range and a selected
display function do not establish an encoding.

```python
from eyepy.io import HeVolWriter

# VOL inputs and E2E imports with supported normalization/registration:
HeVolWriter(volume).write('linear.vol')

# Display-only XML/TIFF inputs; explicitly accept approximate linear values:
HeVolWriter(display_volume, allow_approximate=True).write('approximate.vol')
```

The writer uses linear samples directly when available, preserving float32
values and invalid sentinels. For display-only inputs, it requires
`allow_approximate=True` and inverts the stored display greys (uint8, or float
greys in [0, 1]). Custom visualization settings do not affect either path.
Unknown encodings and E2E inputs without supported conversion metadata are
rejected even when approximation is enabled. Export also requires the usual
VOL patient, examination, localizer and B-scan metadata; this does not establish
compatibility with importing the output into every HEYEX version.

Matching an E2E reconstruction to a paired vendor VOL export is a statement
about that output. Gain clipping, quantization and spatial resampling mean
the conversion is not reversible to the original E2E samples. XML/TIFF display
greys additionally lack the original linear intensities.

`.eye` files persist the raw intensity encoding and the versioned
`bscan_transform` configuration separately from display settings. For older
files without an explicit encoding, loading uses the saved intensity function:
`vol` identifies `vol_linear`, and `e2e` identifies `e2e_raw`. The saved
function continues to control display. `default` and other ambiguous cases
remain `unknown`: an identity function does not establish the data's origin,
and custom functions were historically saved as `default`.

Recognizing E2E samples does not reconstruct missing gain or registration
metadata. These older files retain their fallback display, but cannot provide
VOL-linear data without supported conversion parameters. New files always use
their explicit encoding, even when the display function has been changed.
