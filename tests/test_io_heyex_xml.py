import numpy as np

import eyepy as ep
from eyepy.io import HeXmlReader


def test_hexmlreader():
    reader = HeXmlReader(
        'tests/data/filetypes/heyex_xml/test_volume')
    assert len(reader.meta) > 0


def test_heyex_xml_import():
    data = ep.import_heyex_xml('tests/data/filetypes/heyex_xml/test_volume')
    assert data.shape == (10, 40, 50)
    assert data.localizer.shape == (50, 50)
    assert data.laterality == 'OD'
    assert int(np.mean(data.data)) == 127


def test_heyex_xml_imports_manual_layer_flags():
    data = ep.import_heyex_xml('tests/data/filetypes/heyex_xml/test_volume')

    assert data.layers['RPE'].manual == [True] * data.size_z
    assert data.layers['RPE'].meta['manual'] == [True] * data.size_z
    assert data.layers['BM'].manual == [False] * data.size_z
    assert data[0].layers['RPE'].manual is True
    assert data[0].layers['BM'].manual is False
