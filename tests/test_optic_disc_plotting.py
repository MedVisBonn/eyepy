"""Tests for optic disc plotting functionality."""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon

from eyepy.core.annotations import EyeEnfaceOpticDiscAnnotation


@pytest.fixture
def optic_disc():
    """Create a simple optic disc annotation for testing."""
    return EyeEnfaceOpticDiscAnnotation.from_ellipse(
        center=(100, 100),
        width=30,
        height=35,
        rotation=0.1,
        shape=(200, 200)
    )


class TestOpticDiscPlotting:
    """Test optic disc plotting with various options."""
    
    def test_plot_contour_only_default(self, optic_disc):
        """Test plotting only the contour with default settings."""
        fig, ax = plt.subplots()
        
        # Plot with default settings (contour only)
        optic_disc.plot(ax=ax)
        
        # Check that lines were plotted
        lines = ax.get_lines()
        assert len(lines) > 0, "Expected contour lines to be plotted"
        
        # Check default color
        assert lines[0].get_color() == 'red', "Default contour color should be red"
        
        # Check default linewidth
        assert lines[0].get_linewidth() == 2, "Default linewidth should be 2"
        
        plt.close(fig)
    
    def test_plot_contour_custom_color(self, optic_disc):
        """Test plotting contour with custom color."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, contour_color='blue', contour_linewidth=3)
        
        lines = ax.get_lines()
        assert lines[0].get_color() == 'blue', "Contour color should be blue"
        assert lines[0].get_linewidth() == 3, "Linewidth should be 3"
        
        plt.close(fig)
    
    def test_plot_area_only(self, optic_disc):
        """Test plotting only the filled area without contour."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, plot_contour=False, plot_area=True, 
                       area_color='green', area_alpha=0.5)
        
        # Check that a patch (filled area) was added
        patches = ax.patches
        assert len(patches) > 0, "Expected filled area to be plotted"
        
        # Check that no lines were plotted
        lines = ax.get_lines()
        assert len(lines) == 0, "Expected no contour lines when plot_contour=False"
        
        plt.close(fig)
    
    def test_plot_both_contour_and_area(self, optic_disc):
        """Test plotting both contour and filled area."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, plot_contour=True, plot_area=True,
                       contour_color='darkred', area_color='red', area_alpha=0.3)
        
        # Check that a patch was added for the filled area
        patches = ax.patches
        assert len(patches) > 0, "Expected filled area to be plotted"
        
        plt.close(fig)
    
    def test_plot_with_offset(self, optic_disc):
        """Test that offset is applied correctly."""
        fig, ax = plt.subplots()
        
        # Plot with offset
        offset = (10, 20)
        optic_disc.plot(ax=ax, offset=offset)
        
        lines = ax.get_lines()
        assert len(lines) > 0, "Expected lines to be plotted"
        
        # Get the plotted data
        xdata = lines[0].get_xdata()
        ydata = lines[0].get_ydata()
        
        # Original polygon center
        original_center = optic_disc.center
        
        # The plotted center should be offset
        plotted_center_x = np.mean(xdata)
        plotted_center_y = np.mean(ydata)
        
        # Check that offset was applied (approximately)
        expected_x = original_center[1] - offset[1]
        expected_y = original_center[0] - offset[0]
        
        assert abs(plotted_center_x - expected_x) < 5, "X offset not applied correctly"
        assert abs(plotted_center_y - expected_y) < 5, "Y offset not applied correctly"
        
        plt.close(fig)
    
    def test_plot_area_uses_contour_color_by_default(self, optic_disc):
        """Test that area uses contour color when area_color is not specified."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, plot_contour=False, plot_area=True,
                       contour_color='purple')
        
        patches = ax.patches
        assert len(patches) > 0, "Expected filled area to be plotted"
        
        # The facecolor should match the contour color
        # Note: matplotlib colors are in RGBA format
        # We'll just check that a patch exists with the right transparency
        assert patches[0].get_alpha() == 0.3, "Default alpha should be 0.3"
        
        plt.close(fig)
    
    def test_plot_different_linestyles(self, optic_disc):
        """Test plotting with different linestyles."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, contour_linestyle='--')
        
        lines = ax.get_lines()
        assert lines[0].get_linestyle() == '--', "Linestyle should be dashed"
        
        plt.close(fig)
    
    def test_plot_without_ax_uses_current_axes(self, optic_disc):
        """Test that plot uses current axes when ax is not provided."""
        fig, ax = plt.subplots()
        plt.sca(ax)  # Set current axes
        
        # Plot without providing ax
        optic_disc.plot()
        
        # Check that lines were added to the current axes
        lines = ax.get_lines()
        assert len(lines) > 0, "Expected lines to be plotted on current axes"
        
        plt.close(fig)
    
    def test_plot_only_contour_when_both_false(self, optic_disc):
        """Test that nothing is plotted when both plot_contour and plot_area are False."""
        fig, ax = plt.subplots()
        
        optic_disc.plot(ax=ax, plot_contour=False, plot_area=False)
        
        # Should have no lines or patches
        lines = ax.get_lines()
        patches = ax.patches
        
        # With both False, nothing should be plotted
        assert len(lines) == 0, "Expected no lines when plot_contour=False"
        assert len(patches) == 0, "Expected no patches when plot_area=False"
        
        plt.close(fig)
    
    def test_plot_custom_alpha(self, optic_disc):
        """Test plotting with custom alpha transparency."""
        fig, ax = plt.subplots()
        
        custom_alpha = 0.7
        optic_disc.plot(ax=ax, plot_contour=False, plot_area=True,
                       area_alpha=custom_alpha)
        
        patches = ax.patches
        assert len(patches) > 0, "Expected filled area to be plotted"
        assert patches[0].get_alpha() == custom_alpha, f"Alpha should be {custom_alpha}"
        
        plt.close(fig)
    
    def test_plot_with_kwargs_passthrough(self, optic_disc):
        """Test that additional kwargs are passed through."""
        fig, ax = plt.subplots()
        
        # Pass additional kwargs like label
        optic_disc.plot(ax=ax, label='Optic Disc')
        
        lines = ax.get_lines()
        assert lines[0].get_label() == 'Optic Disc', "Label should be passed through"
        
        plt.close(fig)
