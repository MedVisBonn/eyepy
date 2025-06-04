from __future__ import annotations

import logging
from typing import Optional, Union

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_scalebar(scale: tuple[float, float],
                  scale_unit: str,
                  scale_length: Optional[Union[int, float]] = None,
                  pos: tuple[int, int] = (100, 100),
                  flip_x: bool = False,
                  flip_y: bool = False,
                  color: str = 'white',
                  linewidth: float = 1.5,
                  ax: Optional[plt.Axes] = None,
                  **kwargs: dict) -> None:
    """Plot a scalebar for an image.

    Args:
        scale: tuple of floats (x, y) with the scale in units per pixel. If the `scale_unit` is "px" the scale is ignored.
        scale_unit: unit of the scalebar ("px" or "µm", or "mm")
        scale_length: length of the scalebar in units
        pos: position of the scalebar in pixels
        flip_x: flip the scalebar in x direction
        flip_y: flip the scalebar in y direction
        color: color of the scalebar
        linewidth: linewidth of the scalebar
        ax: matplotlib axis to plot on
        **kwargs: additional keyword arguments passed to ax.plot
    Returns:
        None
    """
    ax = plt.gca() if ax is None else ax

    x, y = pos

    if scale_unit == 'px':
        scale = (1.0, 1.0)

    if scale_length is None:
        if scale_unit == 'px':
            scale_length = 100
        elif scale_unit == 'µm':
            scale_length = 500
        elif scale_unit == 'mm':
            scale_length = 0.5
        else:
            logger.info(
                f'Unknown scale unit: {scale_unit}. Scalebar not plotted.')
            return

    x_start = x
    x_end = x + (scale_length / scale[0])

    y_start = y
    y_end = y - (scale_length / scale[1])

    text_x = x + 8
    text_y = y - 8

    if flip_x:
        x_start = x - (scale_length / scale[0])
        x_end = x
        text_x = x - 50

    if flip_y:
        y_start = y + (scale_length / scale[1])
        y_end = y
        text_y = y + 17

    # Plot horizontal line
    ax.plot([x_start, x_end], [y, y],
            color=color,
            linewidth=linewidth,
            **kwargs)
    # Plot vertical line
    ax.plot([x, x], [y_start, y_end],
            color=color,
            linewidth=linewidth,
            **kwargs)

    ax.text(text_x,
            text_y,
            f'{scale_length}{scale_unit}',
            fontsize=7,
            weight='bold',
            color=color)


def plot_watermark(ax: plt.Axes) -> None:
    """Add a watermark in the lower right corner of a matplotlib axes object.

    Args:
        ax: Axes object
    Returns:
        None
    """
    ax.text(0.98,
            0.02,
            'Visualized with eyepy',
            fontsize=6,
            color='white',
            ha='right',
            va='bottom',
            alpha=0.4,
            transform=ax.transAxes,
            bbox=dict(boxstyle='Round',
                      facecolor='gray',
                      alpha=0.2,
                      linewidth=0))
