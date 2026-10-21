"""Verified Heidelberg normalization and per-B-scan registration arithmetic."""

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
