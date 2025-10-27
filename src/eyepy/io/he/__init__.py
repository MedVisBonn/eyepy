try:
    from .e2e_reader import HeE2eReader
    HAS_E2E_SUPPORT = True
except ImportError:
    HAS_E2E_SUPPORT = False

from .vol_reader import HeVolReader
from .vol_reader import HeVolWriter
from .xml_reader import HeXmlReader

__all__ = ['HeVolReader', 'HeVolWriter', 'HeXmlReader']

if HAS_E2E_SUPPORT:
    __all__.append('HeE2eReader')
