# -*- coding: utf-8 -*-
from .base import bscan_base_spec, oct_base_spec


def oct_spec():
    # Difference with regard to a previous specification
    # "HSF-OCT-103"
    v103_oct = {}
    combined = {**oct_base_spec(), **v103_oct}
    return [(key, *value) for key, value in combined.items()]


def bscan_spec():
    # Difference with regard to a previous specification
    # "HSF-BS-103"
    v103_bscan = {}
    combined = {**bscan_base_spec(), **v103_bscan}
    return [(key, *value) for key, value in combined.items()]
