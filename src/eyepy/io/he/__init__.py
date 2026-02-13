__all__ = ['HeVolReader', 'HeVolWriter', 'HeXmlReader', 'HeE2eReader']

_LAZY_IMPORTS = {
    'HeVolReader': '.vol_reader',
    'HeVolWriter': '.vol_reader',
    'HeXmlReader': '.xml_reader',
    'HeE2eReader': '.e2e_reader',
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
