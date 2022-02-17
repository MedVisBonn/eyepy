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
# x = sns.color_palette("husl", 17)
# color_palette = sns.color_palette(x[::3] + x[1::3] + x[2::3])
# layer_colors = {key: color_palette[value] for key, value in SEG_MAPPING.items()}
layer_colors = defaultdict(lambda: "red")
