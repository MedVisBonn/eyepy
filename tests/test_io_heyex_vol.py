import eyepy as ep
from eyepy.io import HeVolReader
from eyepy.io import HeVolWriter


def test_hexmlreader():
    reader = HeVolReader('tests/data/filetypes/heyex_vol/test_volume.vol')
    assert len(reader.meta) > 0


def test_heyex_vol_import():
    data = ep.import_heyex_vol(
        'tests/data/filetypes/heyex_vol/test_volume.vol')
    assert data.shape == (10, 40, 50)
    assert data.localizer.shape == (50, 50)
    assert data.laterality == 'OD'


#def test_heyex_vol_write_eyevolume(eyevolume, tmp_path):
#    HeVolWriter(eyevolume).write(tmp_path / "test.vol")
#    data = ep.import_heyex_vol(tmp_path / "test.vol")
#    assert data.shape == eyevolume.shape
#    assert data.localizer.shape == eyevolume.localizer.shape
