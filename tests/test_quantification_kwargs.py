
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

plt = pytest.importorskip('matplotlib.pyplot')

import eyepy as ep


def test_plot_quantification_kwargs():
    # Create a dummy volume
    data = np.zeros((10, 10, 10))
    ev = ep.EyeVolume(data=data)

    # Set laterality
    ev.laterality = 'OD'

    # Add a pixel annotation
    ev.add_pixel_annotation(voxel_map=np.zeros((10, 10, 10), dtype=bool), name='drusen')

    # Mock the plot_quantification method of the annotation to check if kwargs are passed
    # We need to mock the annotation object that is stored in ev.volume_maps
    # Access the annotation object
    annotation = ev.volume_maps['drusen']

    # We want to spy on plot_quantification, but it's a method of an object.
    # unittest.mock.patch.object is good for this.
    with patch.object(annotation, 'plot_quantification', wraps=annotation.plot_quantification) as mock_plot:
        quantification_kwargs = {'alpha': 0.8, 'cmap': 'Blues', 'vmin': 0, 'vmax': 10}

        try:
            ev.plot(quantification='drusen', quantification_kwargs=quantification_kwargs)
        except Exception as e:
            pytest.fail(f'Plotting quantification failed with exception: {e}')
        finally:
            plt.close('all')

        # Verify that plot_quantification was called with the expected kwargs
        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args.kwargs

        assert call_kwargs.get('alpha') == 0.8
        assert call_kwargs.get('cmap') == 'Blues'
        assert call_kwargs.get('vmin') == 0
        assert call_kwargs.get('vmax') == 10
        # Also check that standard args are passed
        assert 'region' in call_kwargs
        assert 'ax' in call_kwargs
