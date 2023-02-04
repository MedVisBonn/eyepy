import numpy as np

import eyepy as ep
from eyepy.io import he_xml_reader


def test_hexmlreader():
    reader = he_xml_reader.HeXmlReader(
        "tests/data/filetypes/heyex_xml/test_volume")
    assert len(reader.meta) > 0


def test_heyex_xml_import():
    data = ep.import_heyex_xml("tests/data/filetypes/heyex_xml/test_volume")
    assert data.shape == (10, 40, 50)
    assert data.localizer.shape == (50, 50)
    assert data.laterality == "OD"
    assert int(np.mean(data.data)) == 127
