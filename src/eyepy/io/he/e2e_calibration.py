"""Explicitly approximate optical models and E2E angular registration.

Decoded inputs are distinct from optical constants: neither model below
is a verified vendor equation. Default E2E imports do not select an
optical model.
"""

from dataclasses import asdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class E2EOpticalModel:
    """Reciprocal optical model q = 1 / (a / radius + b + c * focus).

    Args:
        name: Provenance label retained in volume metadata.
        a: Coefficient of inverse corneal radius in millimeters.
        b: Constant denominator term.
        c: Coefficient of focus in diopters.
    """

    name: str
    a: float
    b: float
    c: float

    def mm_per_degree(self, radius_mm: float, focus_diopters: float) -> float:
        """Evaluate the model after validating its inputs and coefficients."""
        values = (radius_mm, focus_diopters, self.a, self.b, self.c)
        if not all(math.isfinite(v) for v in values) or radius_mm <= 0:
            raise ValueError('Optical inputs must be finite and radius positive')
        denominator = self.a / radius_mm + self.b + self.c * focus_diopters
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('Optical model denominator must be finite and positive')
        return 1 / denominator

    def as_dict(self) -> dict:
        return {**asdict(self), 'status': 'approximate',
                'equation': 'q = 1 / (a / radius_mm + b + c * focus_diopters)'}


# Public FoveaAnalysis/Modeling.m, commit
# 5b3cb4a3e5021043f20cd41a87655b1b996f082a, lines 89 and 125.
RESEARCH_OPTICAL_MODEL = E2EOpticalModel(
    'FoveaAnalysis adapted Garway-Heath', 17.21 / 1.02302,
    1.247 / 1.02302, 1 / (17.455 * 1.02302))

# Fit to four effective XML calibration points, radii 6.2/7.7/9.0 mm and
# focus -1.51/-0.49/+0.82/+0.90 D. No independent coordinate-level holdout.
EMPIRICAL_OPTICAL_MODEL = E2EOpticalModel(
    'Four-point E2E/XML fit (2026-10)', 16.828362441469007,
    1.2182834015811586, 0.05588751262262184)


def inverse_localizer_registration(angles: np.ndarray, affine,
                                   localizer_angle_deg: float,
                                   localizer_width: int) -> np.ndarray:
    """Undo the observed Type 10025 mapping on centered angular points.

    Translation is in localizer pixels. No top-left-origin shift is
    applied before the inverse. Verified for one reference/follow-up
    pair, not other scan registrations or B-scan intensity resampling.
    """
    points = np.asarray(angles, dtype=float)
    matrix = np.asarray(affine, dtype=float).reshape(2, 3)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('Expected finite angular points with final dimension 2')
    if (not np.isfinite(matrix).all() or not math.isfinite(localizer_angle_deg)
            or localizer_angle_deg <= 0 or localizer_width <= 0):
        raise ValueError('Invalid localizer registration or angular sampling')
    translation = matrix[:, 2] * localizer_angle_deg / localizer_width
    try:
        return np.linalg.solve(matrix[:, :2],
                               (points - translation).reshape(-1, 2).T).T.reshape(points.shape)
    except np.linalg.LinAlgError as error:
        raise ValueError('Singular localizer registration') from error
