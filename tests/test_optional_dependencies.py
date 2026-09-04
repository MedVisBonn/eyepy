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


def test_require_matplotlib_missing():
    """Test that plotting raises ImportError when matplotlib is not installed."""
    with patch.dict(sys.modules, {'matplotlib': None, 'matplotlib.pyplot': None}):
        from eyepy.core._compat import require_matplotlib

        with pytest.raises(ImportError, match='matplotlib is required for plotting'):
            require_matplotlib('pyplot')


def test_require_matplotlib_available():
    """Test that require_matplotlib returns the module when matplotlib is
    installed."""
    try:
        from eyepy.core._compat import require_matplotlib
        plt = require_matplotlib('pyplot')
        assert hasattr(plt, 'gca')
    except ImportError:
        pytest.skip('matplotlib not installed')


@pytest.mark.parametrize('function_name', ['import_duke_mat', 'import_dukechiu2_mat'])
def test_import_duke_without_scipy(function_name):
    """Duke importers point users to the quant extra when scipy is absent."""
    import eyepy as ep

    with patch.dict(sys.modules, {'scipy': None, 'scipy.io': None}):
        with pytest.raises(ImportError, match=r'pip install eyepy\[quant\]'):
            getattr(ep, function_name)('dummy.mat')


def test_import_retouch_without_itk():
    """RETOUCH importer points users to the itk extra when ITK is absent."""
    import eyepy as ep

    with patch.dict(sys.modules, {'itk': None}):
        with pytest.raises(ImportError, match=r'pip install eyepy\[itk\]'):
            ep.import_retouch('dummy')


def test_e2e_table_without_pandas():
    """E2E tables point users to the pandas extra when pandas is absent."""
    from eyepy.io.he.e2e_reader import E2EFileStructure

    with patch.dict(sys.modules, {'pandas': None}):
        with pytest.raises(ImportError, match=r'pip install eyepy\[pandas\]'):
            E2EFileStructure._get_table(
                object(), {}, structure='E2EFileStructure'
            )
