import logging
from pathlib import Path
import urllib.request
import zipfile

import eyepy as ep

logger = logging.getLogger(__name__)

SAMPLE_DATA = {
    'drusen_patient': (
        'https://uni-bonn.sciebo.de/s/VD8CPAgDKp2EYlm/download',
        ep.import_heyex_xml,
    )
}

EYEPY_DATA_DIR = Path('~/.eyepy/data').expanduser()
if not EYEPY_DATA_DIR.is_dir():
    EYEPY_DATA_DIR.mkdir(parents=True)


def load(name: str) -> ep.EyeVolume:
    """Load sample data.

    Args:
        name: Name of the sample data to load. Currently only "drusen_patient" is supported.

    Returns:
        Sample data with many drusen as EyeVolume object.
    """
    data_dir = EYEPY_DATA_DIR / name
    if not data_dir.is_dir():
        download_path = EYEPY_DATA_DIR / (name + '.zip')
        urllib.request.urlretrieve(SAMPLE_DATA[name][0], download_path)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(EYEPY_DATA_DIR)
        download_path.unlink()

    return SAMPLE_DATA[name][1](EYEPY_DATA_DIR / name / 'metaclean.xml')
