__all__ = ['HeVolReader', 'HeVolWriter', 'HeVolExportContext', 'HeXmlReader', 'HeE2eReader', 'HeE2eWriter', 'HeXmlWriter', 'HeXmlExportContext']

_LAZY_IMPORTS = {
    'HeVolReader': '.vol_reader',
    'HeVolWriter': '.vol_reader',
    'HeVolExportContext': '._vol_export',
    'HeXmlReader': '.xml_reader',
    'HeE2eReader': '.e2e_reader',
    'HeE2eWriter': '.e2e_writer',
    'HeXmlWriter': '.xml_writer',
    'HeXmlExportContext': '._xml_export',
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
