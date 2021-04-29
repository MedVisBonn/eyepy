# -*- coding: utf-8 -*-
from .v103 import bscan_spec as v103_bscan
from .v103 import oct_spec as v103_oct


def HEVOL_VERSIONS(version):
    versions = {"HSF-OCT-103": v103_oct}
    return versions[version]()


def HEVOL_BSCAN_VERSIONS(version):
    versions = {"HSF-BS-103": v103_bscan}
    return versions[version]()
