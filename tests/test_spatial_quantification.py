"""Tests for spatial quantification functionality.

Tests the refactored spatial quantification system with ExtentMetrics
and DirectionalExtent, including per-direction border detection.
"""

import numpy as np
import pytest

from eyepy.quant import AnatomicalOrigin
from eyepy.quant import DirectionalExtent
from eyepy.quant import ExtentMetrics
from eyepy.quant import PolarReference


class TestExtentMetrics:
    """Test ExtentMetrics dataclass."""

    def test_extent_metrics_creation(self):
        """Test creating ExtentMetrics with all fields."""
        metrics = ExtentMetrics(
            midpoint=10.0,
            mean=12.5,
            max=15.0,
            median=12.0,
            std=2.5,
            touches_border=True,
        )

        assert metrics.midpoint == 10.0
        assert metrics.mean == 12.5
        assert metrics.max == 15.0
        assert metrics.median == 12.0
        assert metrics.std == 2.5
        assert metrics.touches_border is True

    def test_extent_metrics_to_dict(self):
        """Test converting ExtentMetrics to dictionary."""
        metrics = ExtentMetrics(
            midpoint=10.0,
            mean=12.5,
            max=15.0,
            median=12.0,
            std=2.5,
            touches_border=False,
        )

        d = metrics.to_dict()

        assert d['midpoint'] == 10.0
        assert d['mean'] == 12.5
        assert d['max'] == 15.0
        assert d['median'] == 12.0
        assert d['std'] == 2.5
        assert d['touches_border'] is False

    def test_extent_metrics_default_border_flag(self):
        """Test that touches_border defaults to False."""
        metrics = ExtentMetrics(
            midpoint=10.0,
            mean=12.5,
            max=15.0,
            median=12.0,
            std=2.5,
        )

        assert metrics.touches_border is False


class TestAnatomicalOrigin:
    """Test AnatomicalOrigin creation and coordinate transformations."""

    def test_from_optic_disc_right_eye(self):
        """Test creating origin from optic disc for right eye."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 100.0),
            laterality='OD',
        )

        assert origin.y == 50.0
        assert origin.x == 100.0
        assert origin.laterality == 'OD'

    def test_from_optic_disc_left_eye(self):
        """Test creating origin from optic disc for left eye."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 100.0),
            laterality='OS',
        )

        assert origin.y == 50.0
        assert origin.x == 100.0
        assert origin.laterality == 'OS'

    def test_from_fovea(self):
        """Test creating origin from fovea."""
        origin = AnatomicalOrigin.from_fovea(
            fovea_center=(60.0, 120.0),
            laterality='OD',
        )

        assert origin.y == 60.0
        assert origin.x == 120.0
        assert origin.laterality == 'OD'

    def test_from_hybrid(self):
        """Test creating hybrid origin (OD x, fovea y)."""
        origin = AnatomicalOrigin.from_hybrid(
            optic_disc_center=(50.0, 100.0),
            fovea_center=(60.0, 120.0),
            laterality='OD',
        )

        # Should use OD x-coordinate and fovea y-coordinate
        assert origin.y == 60.0  # From fovea
        assert origin.x == 100.0  # From optic disc
        assert origin.laterality == 'OD'

    def test_from_custom(self):
        """Test creating custom origin."""
        origin = AnatomicalOrigin.from_custom(
            origin=(75.0, 150.0),
            laterality='OS',
        )

        assert origin.y == 75.0
        assert origin.x == 150.0
        assert origin.laterality == 'OS'

    def test_invalid_laterality_raises_error(self):
        """Test that invalid laterality raises ValueError."""
        with pytest.raises(ValueError, match='Laterality must be OD or OS'):
            AnatomicalOrigin.from_optic_disc(
                optic_disc_center=(50.0, 100.0),
                laterality='INVALID',
            )

    def test_to_cartesian_right_eye(self):
        """Test coordinate transformation to Cartesian for right eye."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )

        # Point to the right of origin (temporally for OD)
        x_cart, y_cart = origin.to_cartesian(50.0, 60.0)
        assert x_cart == -10.0  # Temporal is negative x for OD
        assert y_cart == 0.0

    def test_to_cartesian_left_eye(self):
        """Test coordinate transformation to Cartesian for left eye."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OS',
        )

        # Point to the right of origin (temporally for OS)
        x_cart, y_cart = origin.to_cartesian(50.0, 60.0)
        assert x_cart == 10.0  # Temporal is positive x for OS
        assert y_cart == 0.0


class TestPolarReference:
    """Test PolarReference system."""

    def test_initialization(self):
        """Test PolarReference initialization."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        assert polar_ref.origin == origin

    def test_compute_directional_extent_empty_mask(self):
        """Test computing extent on empty mask returns zeros."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        mask = np.zeros((100, 100), dtype=bool)
        extent = polar_ref.compute_directional_extent(mask)

        # All directions should have zero metrics
        assert extent.temporal.mean == 0.0
        assert extent.temporal.midpoint == 0.0
        assert extent.temporal.max == 0.0
        assert extent.temporal.touches_border is False

        assert extent.superior.mean == 0.0
        assert extent.inferior.mean == 0.0
        assert extent.nasal.mean == 0.0

    def test_compute_directional_extent_centered_square(self):
        """Test computing extent for a centered square region."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        # Create a centered square
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        extent = polar_ref.compute_directional_extent(mask, scale_x=1.0, scale_y=1.0)

        # Check that we have ExtentMetrics for all 8 directions
        assert isinstance(extent.temporal, ExtentMetrics)
        assert isinstance(extent.nasal, ExtentMetrics)
        assert isinstance(extent.superior, ExtentMetrics)
        assert isinstance(extent.inferior, ExtentMetrics)
        assert isinstance(extent.superior_temporal, ExtentMetrics)
        assert isinstance(extent.inferior_temporal, ExtentMetrics)
        assert isinstance(extent.superior_nasal, ExtentMetrics)
        assert isinstance(extent.inferior_nasal, ExtentMetrics)

        # All directions should have positive distances
        assert extent.temporal.mean > 0
        assert extent.nasal.mean > 0
        assert extent.superior.mean > 0
        assert extent.inferior.mean > 0

        # No borders touched for centered square
        assert extent.temporal.touches_border is False
        assert extent.superior.touches_border is False
        assert extent.inferior.touches_border is False
        assert extent.nasal.touches_border is False

    def test_compute_directional_extent_with_scaling(self):
        """Test that scaling affects computed distances."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        # Compute with scale=1
        extent1 = polar_ref.compute_directional_extent(mask, scale_x=1.0, scale_y=1.0)

        # Compute with scale=2
        extent2 = polar_ref.compute_directional_extent(mask, scale_x=2.0, scale_y=2.0)

        # Distances should be doubled
        assert extent2.temporal.mean == pytest.approx(extent1.temporal.mean * 2.0, rel=0.01)
        assert extent2.superior.mean == pytest.approx(extent1.superior.mean * 2.0, rel=0.01)


class TestDirectionalExtent:
    """Test DirectionalExtent dataclass."""

    def test_directional_extent_creation(self):
        """Test creating DirectionalExtent with all directions."""
        metrics = ExtentMetrics(
            midpoint=10.0, mean=12.0, max=15.0, median=11.0, std=2.0, touches_border=False
        )

        extent = DirectionalExtent(
            temporal=metrics,
            nasal=metrics,
            superior=metrics,
            inferior=metrics,
            superior_temporal=metrics,
            inferior_temporal=metrics,
            superior_nasal=metrics,
            inferior_nasal=metrics,
        )

        assert extent.temporal == metrics
        assert extent.nasal == metrics
        assert extent.superior == metrics
        assert extent.inferior == metrics
        assert extent.superior_temporal == metrics
        assert extent.inferior_temporal == metrics
        assert extent.superior_nasal == metrics
        assert extent.inferior_nasal == metrics

    def test_directional_extent_to_dict(self):
        """Test converting DirectionalExtent to nested dictionary."""
        metrics = ExtentMetrics(
            midpoint=10.0, mean=12.0, max=15.0, median=11.0, std=2.0, touches_border=True
        )

        extent = DirectionalExtent(
            temporal=metrics,
            nasal=metrics,
            superior=metrics,
            inferior=metrics,
            superior_temporal=metrics,
            inferior_temporal=metrics,
            superior_nasal=metrics,
            inferior_nasal=metrics,
        )

        d = extent.to_dict()

        # Check nested structure
        assert 'temporal' in d
        assert 'superior' in d
        assert isinstance(d['temporal'], dict)
        assert d['temporal']['mean'] == 12.0
        assert d['temporal']['midpoint'] == 10.0
        assert d['temporal']['touches_border'] is True

        # Check all 8 directions present
        assert set(d.keys()) == {
            'temporal', 'nasal', 'superior', 'inferior',
            'superior_temporal', 'inferior_temporal',
            'superior_nasal', 'inferior_nasal'
        }


class TestBorderDetection:
    """Test per-direction border detection."""

    def test_region_touching_top_border(self):
        """Test that superior direction detects top border."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        # Region extending to top edge
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:50, 10:90] = True

        extent = polar_ref.compute_directional_extent(mask)

        # Superior should touch border
        assert extent.superior.touches_border is True
        # Inferior should not
        assert extent.inferior.touches_border is False

        # Superior-nasal and Superior-temporal should touch border
        assert extent.superior_nasal.touches_border is True
        assert extent.superior_temporal.touches_border is True
        # Inferior-nasal and Inferior-temporal should not touch border
        assert extent.inferior_nasal.touches_border is False
        assert extent.inferior_temporal.touches_border is False

    def test_region_touching_bottom_border(self):
        """Test that inferior direction detects bottom border."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        # Region extending to bottom edge
        mask = np.zeros((100, 100), dtype=bool)
        mask[50:100, 30:70] = True

        extent = polar_ref.compute_directional_extent(mask)

        # Inferior should touch border
        assert extent.inferior.touches_border is True
        # Superior should not
        assert extent.superior.touches_border is False

    def test_region_in_corner_touches_multiple_borders(self):
        """Test that corner region detects multiple border touches."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        # Region in top-left corner
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:50, 0:50] = True

        extent = polar_ref.compute_directional_extent(mask)

        # Multiple directions might touch borders
        assert extent.superior.touches_border is True
        assert extent.nasal.touches_border is False
        assert extent.superior_nasal.touches_border is False
        assert extent.superior_temporal.touches_border is True
        assert extent.inferior_nasal.touches_border is False
        assert extent.inferior_temporal.touches_border is False


class TestStatisticalMetrics:
    """Test that all statistical metrics are computed correctly."""

    def test_all_metrics_computed(self):
        """Test that all metrics are present and non-zero for valid region."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        extent = polar_ref.compute_directional_extent(mask)

        # Check temporal direction has all metrics
        assert extent.temporal.midpoint > 0
        assert extent.temporal.mean > 0
        assert extent.temporal.max > 0
        assert extent.temporal.median > 0
        assert extent.temporal.std >= 0  # std can be 0 for uniform distances

    def test_max_greater_than_or_equal_to_mean(self):
        """Test that max is always >= mean."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        extent = polar_ref.compute_directional_extent(mask)

        # Check for all directions
        for direction in ['temporal', 'nasal', 'superior', 'inferior',
                          'superior_temporal', 'inferior_temporal',
                          'superior_nasal', 'inferior_nasal']:
            metrics = getattr(extent, direction)
            assert metrics.max >= metrics.mean


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_pixel_mask(self):
        """Test computing extent for single pixel."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 60] = True  # Single pixel

        extent = polar_ref.compute_directional_extent(mask)

        # Should not crash and should return valid ExtentMetrics
        assert isinstance(extent, DirectionalExtent)
        assert extent.temporal.midpoint >= 0  # Single pixel may have 0 extent    def test_asymmetric_region(self):
        """Test that asymmetric regions have different directional extents."""
        origin = AnatomicalOrigin.from_optic_disc(
            optic_disc_center=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref = PolarReference(origin)

        # Create asymmetric region (wider in horizontal direction)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 20:80] = True  # 20 pixels tall, 60 pixels wide

        extent = polar_ref.compute_directional_extent(mask)

        # Temporal/nasal should be larger than superior/inferior
        horizontal_extent = (extent.temporal.mean + extent.nasal.mean) / 2
        vertical_extent = (extent.superior.mean + extent.inferior.mean) / 2

        assert horizontal_extent > vertical_extent

    def test_different_origins_give_different_results(self):
        """Test that different origins produce different extent measurements."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        # Origin at center
        origin1 = AnatomicalOrigin.from_custom(
            origin=(50.0, 50.0),
            laterality='OD',
        )
        polar_ref1 = PolarReference(origin1)
        extent1 = polar_ref1.compute_directional_extent(mask)

        # Origin offset
        origin2 = AnatomicalOrigin.from_custom(
            origin=(40.0, 40.0),
            laterality='OD',
        )
        polar_ref2 = PolarReference(origin2)
        extent2 = polar_ref2.compute_directional_extent(mask)

        # Results should differ
        assert extent1.temporal.mean != extent2.temporal.mean
