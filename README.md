# eyepy

[![Documentation](https://img.shields.io/badge/docs-eyepy-blue)](https://MedVisBonn.github.io/eyepy)
[![PyPI version](https://badge.fury.io/py/eyepie.svg)](https://badge.fury.io/py/eyepie)
[![DOI](https://zenodo.org/badge/292547201.svg)](https://zenodo.org/badge/latestdoi/292547201)

The `eyepy` python package provides a simple interface to import and process OCT volumes. Everything you import with one of our import functions becomes an `EyeVolume` object which provides a unified interface to the data. The `EyeVolume` object provides methods to plot the localizer image and B-scans as well as to compute and plot quantifications of voxel annotations such as drusen. Check out the [documentation](https://MedVisBonn.github.io/eyepy), especially the [Cookbook](https://medvisbonn.github.io/eyepy/cookbook/) chapter, for more information.

## Features

* Import HEYEY E2E, VOL and XML exports
* Import B-Scans from a folder
* Import public [AMD Dataset from Duke University](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)
* Import data of the [RETOUCH Challenge](https://retouch.grand-challenge.org/).
* Compute Drusen voxel annotation from BM and RPE layer segmentations
* Quantify voxel annotations on a customizable circular grid
* Plot annotated localizer
* Plot annotated B-scans
* Save and load EyeVolume objects

## Getting Started

### Installation
To install the latest version of eyepy run `pip install -U eyepie`. It is `eyepie` and not `eyepy` for installation with pip.

When you don't hava a supported OCT volume at hand you can check out our sample dataset to get familiar with `eyepy`.
```python
import eyepy as ep
# Import HEYEX XML export
ev = ep.data.load("drusen_patient")
```

# Related Projects:

+ [OCT-Converter](https://github.com/marksgraham/OCT-Converter): Extract raw optical coherence tomography (OCT) and fundus data from proprietary file formats. (.fds/.fda/.e2e/.img/.oct/.dcm)
+ [eyelab](https://github.com/MedVisBonn/eyelab): A GUI for annotation of OCT data based on eyepy
+ Projects by the [Translational Neuroimaging Laboratory](https://github.com/neurodial)
  + [LibOctData](https://github.com/neurodial/LibOctData)
  + [LibE2E](https://github.com/neurodial/LibE2E)
  + [OCT-Marker](https://github.com/neurodial/OCT-Marker)
+ [UOCTE](https://github.com/TSchlosser13/UOCTE) Unofficial continuation of https://bitbucket.org/uocte/uocte
+ [OCTAnnotate](https://github.com/krzyk87/OCTAnnotate)
+ [heyexReader](https://github.com/ayl/heyexReader)
+ [OCTExplorer](https://www.iibi.uiowa.edu/oct-reference) Iowa Reference Algorithm
