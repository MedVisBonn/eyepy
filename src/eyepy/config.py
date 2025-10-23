from collections import defaultdict

# Plotting config

# Line Style for Layers in B-Scan
layer_kwargs = {'linewidth': 1, 'linestyle': '-'}
slab_kwargs = {'alpha': 0.5}
area_kwargs = {'alpha': 0.5}
ascan_kwargs = {'alpha': 0.5}

# Line Style for B-Scan positions on Slo
line_kwargs = {'linewidth': 0.2, 'linestyle': '-', 'color': 'limegreen'}

# Colors for different Layers
_layer_colors = {
    'BM': 'F77189',
    'RPE': '97A431',
    'PR1': '36ADA4',
    'EZ': '36ADA4',
    'PR2': 'A48CF4',
    'IZ': 'A48CF4',
    'ELM': 'E68332',
    'ONL': '50B131',
    'OPL': '38AABF',
    'INL': 'E866F4',
    'IPL': 'BB9832',
    'GCL': '34AF84',
    'RNFL': '3BA3EC',
    'NFL': '3BA3EC',
    'ILM': 'F668C2',
}

layer_colors = defaultdict(lambda: 'FF0000', **_layer_colors)

# Colors for different Slabs
_slab_colors = {
    'NFLVP': '00598C',
    'SVP': '0057E5',
    'ICP': '41B6E6',
    'DCP': '3EA908',
    'SVC': 'F0CC2E',
    'DVC': 'EC894D',
    'AVAC': 'E50000',
    'RET': 'E641B6',
}

slab_colors = defaultdict(lambda: 'FBB2C4', **_slab_colors)

# Colors for different Area annotations
_area_colors = {
}

area_colors = defaultdict(lambda: 'FF0000', **_area_colors)
