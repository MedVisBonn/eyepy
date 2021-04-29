# -*- coding: utf-8 -*-
from .base import bscan_base_spec, oct_base_spec


def oct_spec():
    # Difference with regard to a previous specification
    v69530_oct = {}
    combined = {**oct_base_spec(), **v69530_oct}
    return [(key, *value) for key, value in combined.items()]


def bscan_spec():
    # Difference with regard to a previous specification
    v69530_bscan = {}
    combined = {**bscan_base_spec(), **v69530_bscan}
    return [(key, *value) for key, value in combined.items()]
