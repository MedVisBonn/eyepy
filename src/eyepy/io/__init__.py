"""IO module for reading ophthalmic image formats."""

_IMPORT_FUNCTIONS = [
    'import_bscan_folder',
    'import_duke_mat',
    'import_dukechiu2_mat',
    'import_heyex_angio_vol',
    'import_heyex_e2e',
    'import_heyex_vol',
    'import_heyex_xml',
    'import_retouch',
    'import_topcon_fda',
]

__all__ = _IMPORT_FUNCTIONS + ['HeVolReader', 'HeVolWriter', 'HeXmlReader', 'HeE2eReader']


def __dir__():
    return __all__


def __getattr__(name):
    if name in _IMPORT_FUNCTIONS:
        from . import import_functions
        return getattr(import_functions, name)
    if name in ('HeVolReader', 'HeVolWriter', 'HeXmlReader', 'HeE2eReader'):
        from . import he
        return getattr(he, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
