# -*- coding: utf-8 -*-
from .v61240 import bscan_spec as v61240_bscan
from .v61240 import oct_spec as v61240_oct
from .v69530 import bscan_spec as v69530_bscan
from .v69530 import oct_spec as v69530_oct


def HEXML_VERSIONS(version):
    versions = {"6.12.4.0": v61240_oct, "6.9.53.0": v69530_oct}
    return versions[version]()


def HEXML_BSCAN_VERSIONS(version):
    versions = {"6.12.4.0": v61240_bscan, "6.9.53.0": v69530_bscan}
    return versions[version]()
