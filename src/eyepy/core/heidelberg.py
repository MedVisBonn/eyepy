"""Heidelberg image normalization and registration arithmetic."""

from __future__ import annotations

import numpy as np


def _truncate_fraction(data: np.ndarray) -> np.ndarray:
    """Positive float32: keep ten of the 23 fraction bits, without rounding."""
    return (data.astype('<f4').view('<u4') & np.uint32(0xffffe000)).view('<f4')


def _fma32(a, b, c):
    """Multiply and add before rounding once to float32."""
    return np.float32(
        np.asarray(a, dtype=float) * np.asarray(b, dtype=float) +
        np.asarray(c, dtype=float))


def _lerp(a, b, fraction):
    """Preserve subtraction precision, then round each interpolation stage."""
    return np.float32(a.astype(float) + (b.astype(float) - a) * fraction)


class HeidelbergLocalizerTransform:
    """Type10025 output-to-source affine in centered localizer pixels.

    This map is independent of the Type10012 B-scan image-plane maps.
    Pixel resampling uses bilinear interpolation followed by truncation
    to uint8; exact HEYEX interpolation arithmetic is not fully
    reproduced.
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float32, copy=True)
        if self.matrix.shape != (2, 3) or not np.all(np.isfinite(self.matrix)):
            raise ValueError('Expected a finite 2x3 E2E localizer affine')
        if np.linalg.det(self.matrix[:, :2]) == 0:
            raise ValueError('The E2E localizer affine is singular')
        self.matrix.setflags(write=False)

    def as_dict(self):
        return dict(type='heidelberg_localizer', matrix=self.matrix.tolist())

    @classmethod
    def from_dict(cls, config):
        if config['type'] != 'heidelberg_localizer':
            raise ValueError(f"Unknown localizer transform: {config['type']}")
        return cls(config['matrix'])

    def source_to_output(self, points, shape):
        """Move raw-localizer (x, y) positions into the registered frame."""
        points = np.asarray(points, dtype=float)
        center = np.asarray(shape[::-1], dtype=float) / 2
        shifted = points - center - self.matrix[:, 2]
        return np.linalg.solve(self.matrix[:, :2].astype(float),
                               shifted.reshape(-1, 2).T).T.reshape(points.shape) + center

    def __call__(self, data):
        """Resample a uint8 IR localizer without modifying the source image."""
        from scipy.ndimage import map_coordinates

        if data.ndim != 2 or data.dtype != np.uint8:
            raise ValueError('E2E localizer registration requires a 2D uint8 image')
        if np.array_equal(self.matrix, [[1, 0, 0], [0, 1, 0]]):
            return data.copy()
        height, width = data.shape
        y, x = np.indices(data.shape, dtype=float)
        a, b, tx, c, d, ty = self.matrix.astype(float).ravel()
        source_x = a * (x - width / 2) + b * (y - height / 2) + tx + width / 2
        source_y = c * (x - width / 2) + d * (y - height / 2) + ty + height / 2
        inside = ((source_x >= -.5) & (source_x < width - .5) &
                  (source_y >= -.5) & (source_y < height - .5))
        sampled = map_coordinates(data.astype(float), [source_y, source_x],
                                  order=1, mode='nearest')
        result = np.floor(sampled).astype(np.uint8)
        result[~inside] = 0
        return result


class HeidelbergBscanTransform:
    """Lazy E2E normalization followed by registered linear sampling.

    ``gains[i]`` is Type10004.float32@32 /
    Type10010.float32@(16+4*aktImage). ``matrices[i]`` contains the six
    Type10012 coefficients in (x, y) order. Only the observed a=d=1, b=0
    affine family is supported. No gain or geometric parameter is
    fitted. See docs/formats/he_e2e.md.
    """

    def __init__(self, gains, matrices):
        self.gains = np.array(gains, dtype=np.float64, copy=True)
        self.matrices = np.array(matrices, dtype=np.float32, copy=True)
        if self.gains.ndim != 1 or self.matrices.shape != (len(
                self.gains), 2, 3):
            raise ValueError('Expected one gain and 2x3 matrix per B-scan')
        if not np.all(np.isfinite(self.gains) & (self.gains > 0)):
            raise ValueError(
                'B-scan normalization gains must be finite and positive')
        if not np.all(np.isfinite(self.matrices)):
            raise ValueError('B-scan registration coefficients must be finite')
        if not (np.all(self.matrices[:, 0, 0] == 1) and
                np.all(self.matrices[:, 0, 1] == 0) and
                np.all(self.matrices[:, 1, 1] == 1)):
            raise ValueError('Unverified E2E affine: expected a=d=1 and b=0')
        self.gains.setflags(write=False)
        self.matrices.setflags(write=False)

    def as_dict(self):
        """JSON-compatible configuration, persisted with the original samples."""
        return dict(type='heidelberg_e2e',
                    gains=self.gains.tolist(),
                    matrices=self.matrices.tolist())

    @classmethod
    def from_dict(cls, config):
        if config['type'] != 'heidelberg_e2e':
            raise ValueError(f"Unknown B-scan transform: {config['type']}")
        return cls(config['gains'], config['matrices'])

    def normalize(self, data: np.ndarray, index: int) -> np.ndarray:
        """Prepare one B-scan without modifying its original decoded samples."""
        # 0xFFFF decodes as 1.9990234375; HEYEX fills this raw invalid code
        # with zero before interpolation. Retain the code in raw_data.
        pixels = np.asarray(data, dtype=np.float32)
        pixels = np.where(pixels == 1.9990234375, 0, pixels)
        return _truncate_fraction(
            np.clip(pixels * np.float32(self.gains[index]), 0, 1))

    def register(self, data: np.ndarray, index: int) -> np.ndarray:
        """Register an already normalized B-scan, returning linear VOL values."""
        height, width = data.shape
        _, _, tx, shear, _, ty = self.matrices[index].ravel()
        y, x = np.indices((height, width), dtype=np.float32)
        offset_y = float(ty) - float(shear) * (width / 2)
        base_y = np.float32(y.astype(float) + offset_y)
        source_y = _fma32(shear, x, base_y)
        source_x = x + tx
        inside = ((source_y >= -.5) & (source_y < height - .5) &
                  (source_x >= -.5) & (source_x < width - .5))
        source_y = np.clip(source_y, 0, height - 1)
        source_x = np.clip(source_x, 0, width - 1)
        y0 = np.floor(source_y).astype(np.int32)
        x0 = np.floor(source_x).astype(np.int32)
        fy = source_y - y0.astype(np.float32)
        fx = source_x - x0.astype(np.float32)
        y1 = np.minimum(y0 + 1, height - 1)
        x1 = np.minimum(x0 + 1, width - 1)
        top = _lerp(data[y0, x0], data[y0, x1], fx)
        bottom = _lerp(data[y1, x0], data[y1, x1], fx)
        registered = _truncate_fraction(_lerp(top, bottom, fy))
        registered[~inside] = np.finfo(np.float32).max
        return registered

    def __call__(self,
                 data: np.ndarray,
                 index: int | None = None) -> np.ndarray:
        """Normalize, truncate, register, then truncate; never apply display."""
        if index is not None:
            return self.register(self.normalize(data, index), index)
        if data.ndim != 3 or data.shape[0] != len(self.gains):
            raise ValueError('Volume does not match the B-scan transform')
        result = np.empty(data.shape, dtype=np.float32)
        for i, bscan in enumerate(data):
            result[i] = self(bscan, i)
        return result

    def transform_layers(self, heights: np.ndarray) -> np.ndarray:
        """Map raw layer heights into registered coordinates, preserving NaNs.

        Heights are continuous coordinates; image intensity truncation
        does not apply. Missing horizontal support remains NaN.
        """
        if heights.shape[0] != len(self.gains):
            raise ValueError('Layer does not match the B-scan transform')
        width = heights.shape[1]
        x = np.arange(width, dtype=float)
        result = np.empty(heights.shape, dtype=float)
        for i, matrix in enumerate(self.matrices):
            tx, shear, ty = float(matrix[0,
                                         2]), float(matrix[1,
                                                           0]), float(matrix[1,
                                                                             2])
            result[i] = np.interp(x + tx,
                                  x,
                                  heights[i],
                                  left=np.nan,
                                  right=np.nan)
            result[i] -= shear * (x - width / 2) + ty
        return result
