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
`to_vol_intensity()` chooses an intensity in the middle of a display bin;
it cannot recover intensities lost during conversion to 8-bit greys.
