# -*- coding: utf-8 -*-
from collections import defaultdict

# Plotting config

# Line Style for Layers in B-Scan
layer_kwargs = {"linewidth": 1, "linestyle": "-"}
area_kwargs = {"alpha": 0.5}
ascan_kwargs = {"alpha": 0.5}

# Line Style for B-Scan positions on Slo
line_kwargs = {"linewidth": 0.2, "linestyle": "-", "color": "green"}

# Colors for different Layers
_layer_colors = {
    "BM": "F77189",
    "RPE": "97A431",
    "PR1": "36ADA4",
    "EZ": "36ADA4",
    "PR2": "A48CF4",
    "IZ": "A48CF4",
    "ELM": "E68332",
    "ONL": "50B131",
    "OPL": "38AABF",
    "INL": "E866F4",
    "IPL": "BB9832",
    "GCL": "34AF84",
    "RNFL": "3BA3EC",
    "NFL": "3BA3EC",
    "ILM": "F668C2",
}

layer_colors = defaultdict(lambda: "FF0000", **_layer_colors)
