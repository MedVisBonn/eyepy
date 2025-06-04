import logging
from pathlib import Path
import urllib.request
import zipfile

import eyepy as ep

logger = logging.getLogger(__name__)

SAMPLE_DATA = {
    'drusen_patient': (
        'https://github.com/MedVisBonn/eyepydata/releases/download/v1.0.0/drusen_patient.zip',
        ep.import_heyex_xml,
    ),
    'healthy_OS': (
        'https://github.com/MedVisBonn/eyepydata/releases/download/v1.0.0/healthy_OS.vol',
        ep.import_heyex_vol,
    ),
    'healthy_OD': (
        'https://github.com/MedVisBonn/eyepydata/releases/download/v1.0.0/healthy_OD.vol',
        ep.import_heyex_vol,
    ),
    'healthy_OS_Angio': (
        'https://github.com/MedVisBonn/eyepydata/releases/download/v1.0.0/healthy_OS_Angio.vol',
        ep.import_heyex_angio_vol,
    ),
    'healthy_OD_Angio': (
        'https://github.com/MedVisBonn/eyepydata/releases/download/v1.0.0/healthy_OD_Angio.vol',
        ep.import_heyex_angio_vol,
    ),
}

EYEPY_DATA_DIR = Path('~/.eyepy/data').expanduser()
if not EYEPY_DATA_DIR.is_dir():
    EYEPY_DATA_DIR.mkdir(parents=True)


def load(name: str) -> ep.EyeVolume:
    """Load sample data.

    Args:
        name: Name of the sample data to load.
            Available options are:
            - 'drusen_patient': Sample data with many drusen.
            - 'healthy_OS': Healthy left eye volume.
            - 'healthy_OD': Healthy right eye volume.
            - 'healthy_OS_Angio': Healthy left eye OCT Angiography.
            - 'healthy_OD_Angio': Healthy right eye OCT Angiography.

    Returns:
        ep.EyeVolume: The loaded eye volume data.
    """
    url, import_func = SAMPLE_DATA[name]
    file_ext = Path(url).suffix
    data_path = EYEPY_DATA_DIR / name

    if file_ext == '.zip':
        # ZIP file: extract if not already extracted
        if not data_path.is_dir():
            download_path = EYEPY_DATA_DIR / (name + '.zip')
            urllib.request.urlretrieve(url, download_path)
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(EYEPY_DATA_DIR)
            download_path.unlink()
        # Assume XML inside extracted folder for zip samples
        return import_func(data_path / 'metaclean.xml')
    else:
        # Direct file (e.g., .vol): download if not present
        if not data_path.exists():
            urllib.request.urlretrieve(url, data_path)
        return import_func(data_path)
