"""Test optional dependency handling."""
import sys
from unittest.mock import patch

import pytest


def test_import_topcon_fda_without_oct_converter():
    """Test that import_topcon_fda raises ImportError when oct-converter is not
    installed."""
    # Temporarily hide oct_converter from imports
    with patch.dict(sys.modules, {'oct_converter': None, 'oct_converter.readers': None}):
        # Re-import the module to trigger the ImportError in the try-except block
        import importlib

        import eyepy.io
        importlib.reload(eyepy.io)

        # Now test that calling the function raises the expected error
        with pytest.raises(ImportError, match='oct-converter is required to read FDA files'):
            eyepy.io.import_topcon_fda('dummy_path.fda')


def test_import_topcon_fda_with_oct_converter():
    """Test that import_topcon_fda works when oct-converter is installed."""
    # Import eyepy fresh to check current state
    import eyepy as ep

    # Try to call the function - if oct-converter is installed, we should get
    # a FileNotFoundError or similar, not an ImportError about missing dependency
    try:
        ep.import_topcon_fda('non_existent_file.fda')
        # If we get here without an error, something is wrong
        assert False, 'Expected an error for non-existent file'
    except ImportError as e:
        # If we get ImportError, oct-converter is not installed
        # This is expected if the package wasn't installed with [fda] extra
        assert 'oct-converter is required' in str(e)
    except FileNotFoundError:
        # If we get FileNotFoundError, oct-converter IS installed
        # This means the dependency check passed successfully
        pass
    except Exception as e:
        # Any other error means oct-converter is installed but file processing failed
        # This is also fine - it means the dependency check passed
        assert 'oct-converter is required' not in str(e)
