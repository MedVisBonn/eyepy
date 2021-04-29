# -*- coding: utf-8 -*-
import seaborn as sns

# Plotting config

# PR1 and EZ map to 14 and PR2 and IZ map to 15. Hence both names can be used
# to access the same data
SEG_MAPPING = {
    "ILM": 0,
    "BM": 1,
    "RNFL": 2,
    "NFL": 2,
    "GCL": 3,
    "IPL": 4,
    "INL": 5,
    "OPL": 6,
    "ONL": 7,
    "ELM": 8,
    "IOS": 9,
    "OPT": 10,
    "CHO": 11,
    "VIT": 12,
    "ANT": 13,
    "EZ": 14,
    "PR1": 14,
    "IZ": 15,
    "PR2": 15,
    "RPE": 16,
}

SAVE_DRUSEN = False

# Line Style for Layers in B-Scan
layers_kwargs = {"linewidth": 1, "linestyle": "-"}

# Line Style for B-Scan positions on Slo
line_kwargs = {"linewidth": 0.3, "linestyle": "-"}

# Colors for different Layers
x = sns.color_palette("husl", 17)
color_palette = sns.color_palette(x[::3] + x[1::3] + x[2::3])
layers_color = {key: color_palette[value] for key, value in SEG_MAPPING.items()}
