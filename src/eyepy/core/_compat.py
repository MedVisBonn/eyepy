"""Compatibility helpers for optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def require_matplotlib(submodule: str = 'pyplot') -> ModuleType:
    """Import and return a matplotlib submodule, or raise a clear error.

    Args:
        submodule: The matplotlib submodule to import (e.g. 'pyplot', 'colors')

    Returns:
        The imported module

    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        return importlib.import_module(f'matplotlib.{submodule}')
    except ImportError:
        raise ImportError(
            'matplotlib is required for plotting. '
            'Install it with: pip install eyepy[plot]'
        ) from None
