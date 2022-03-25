# -*- coding: utf-8 -*-
import logging
import warnings

from .base import bscan_base_spec, oct_base_spec
from .v61240 import bscan_spec as v61240_bscan
from .v61240 import oct_spec as v61240_oct
from .v69530 import bscan_spec as v69530_bscan
from .v69530 import oct_spec as v69530_oct

logger = logging.getLogger("eyepy.io.heyex")


def HEXML_VERSIONS(version):
    try:
        versions = {"6.12.4.0": v61240_oct, "6.9.53.0": v69530_oct}
        return versions[version]()
    except KeyError:
        logger.warning("The XML export version is not known.")
        return [(key, *value) for key, value in oct_base_spec().items()]


def HEXML_BSCAN_VERSIONS(version):
    try:
        versions = {"6.12.4.0": v61240_bscan, "6.9.53.0": v69530_bscan}
        return versions[version]()
    except KeyError:
        # logger.warning("The XML export version is not known.")
        return [(key, *value) for key, value in bscan_base_spec().items()]
