"""The eyepy top.

+ [EyeVolume][eyepy.core.eyevolume.EyeVolume]
+ [EyeBscan][eyepy.core.eyebscan.EyeBscan]
+ [EyeEnface][eyepy.core.eyeenface.EyeEnface]

+ [EyeBscanMeta][eyepy.core.eyemeta.EyeBscanMeta]
+ [EyeEnfaceMeta][eyepy.core.eyemeta.EyeEnfaceMeta]
+ [EyeVolumeMeta][eyepy.core.eyemeta.EyeVolumeMeta]

+ [EyeVolumeLayerAnnotation][eyepy.core.annotations.EyeVolumeLayerAnnotation]
+ [EyeBscanLayerAnnotation][eyepy.core.annotations.EyeBscanLayerAnnotation]
+ [EyeBscanSlabAnnotation][eyepy.core.annotations.EyeBscanSlabAnnotation]

+ [EyeEnfacePixelAnnotation][eyepy.core.annotations.EyeEnfacePixelAnnotation]

+ [EyeVolumePixelAnnotation][eyepy.core.annotations.EyeVolumePixelAnnotation]

+ [EyeVolumeSlabAnnotation][eyepy.core.annotations.EyeVolumeSlabAnnotation]

+ [PolygonAnnotation][eyepy.core.annotations.PolygonAnnotation]

+ [EyeEnfaceOpticDiscAnnotation][eyepy.core.annotations.EyeEnfaceOpticDiscAnnotation]

+ [EyeEnfaceFoveaAnnotation][eyepy.core.annotations.EyeEnfaceFoveaAnnotation]
"""
# isort: skip_file

__author__ = """Olivier Morelle"""
__email__ = 'oli4morelle@gmail.com'
__version__ = '0.20.1'

from typing import TYPE_CHECKING

from eyepy.core import (
    EyeBscan,
    EyeBscanLayerAnnotation,
    EyeBscanMeta,
    EyeBscanSlabAnnotation,
    EyeEnface,
    EyeEnfaceFoveaAnnotation,
    EyeEnfaceMeta,
    EyeEnfaceOpticDiscAnnotation,
    EyeEnfacePixelAnnotation,
    EyeVolume,
    EyeVolumeLayerAnnotation,
    EyeVolumeMeta,
    EyeVolumePixelAnnotation,
    EyeVolumeSlabAnnotation,
    PolygonAnnotation,
)
from eyepy.core import annotations

if TYPE_CHECKING:
    from eyepy.io.import_functions import (
        import_bscan_folder,
        import_duke_mat,
        import_dukechiu2_mat,
        import_heyex_angio_vol,
        import_heyex_e2e,
        import_heyex_vol,
        import_heyex_xml,
        import_retouch,
        import_topcon_fda,
    )
    from . import data, io, quant


_LAZY_MAPPING = {
    'import_bscan_folder': 'eyepy.io.import_functions',
    'import_duke_mat': 'eyepy.io.import_functions',
    'import_dukechiu2_mat': 'eyepy.io.import_functions',
    'import_heyex_e2e': 'eyepy.io.import_functions',
    'import_heyex_vol': 'eyepy.io.import_functions',
    'import_heyex_angio_vol': 'eyepy.io.import_functions',
    'import_heyex_xml': 'eyepy.io.import_functions',
    'import_retouch': 'eyepy.io.import_functions',
    'import_topcon_fda': 'eyepy.io.import_functions',
}

__all__ = [
    'EyeBscan',
    'EyeBscanLayerAnnotation',
    'EyeBscanMeta',
    'EyeBscanSlabAnnotation',
    'EyeEnface',
    'EyeEnfaceFoveaAnnotation',
    'EyeEnfaceMeta',
    'EyeEnfaceOpticDiscAnnotation',
    'EyeEnfacePixelAnnotation',
    'EyeVolume',
    'EyeVolumeLayerAnnotation',
    'EyeVolumeMeta',
    'EyeVolumePixelAnnotation',
    'EyeVolumeSlabAnnotation',
    'PolygonAnnotation',
    'annotations',
    'data',
    'io',
    'quant',
] + list(_LAZY_MAPPING.keys())


def __dir__():
    return __all__



def __getattr__(name):
    if name in _LAZY_MAPPING:
        import importlib
        module_path = _LAZY_MAPPING[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    if name in ('data', 'io', 'quant'):
        import importlib
        return importlib.import_module(f'.{name}', __name__)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
