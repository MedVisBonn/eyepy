<h1 align="center">eyepy</h1>
<p align="center">
Use Python to import, analyse and visualize retinal imaging data.
</p>

![header_gif](https://user-images.githubusercontent.com/5720058/228815448-4b561246-dac9-4f8f-abde-e0dd5457a72b.gif)

[![Documentation](https://img.shields.io/badge/docs-eyepy-blue)](https://MedVisBonn.github.io/eyepy)
[![PyPI version](https://badge.fury.io/py/eyepy.svg)](https://badge.fury.io/py/eyepy)
[![DOI](https://zenodo.org/badge/292547201.svg)](https://zenodo.org/badge/latestdoi/292547201)


The `eyepy` python package provides a simple interface to import and process OCT volumes. Everything you import with one of our import functions becomes an `EyeVolume` object which provides a unified interface to the data. The `EyeVolume` object provides methods to plot the localizer (fundus) image and B-scans as well as to compute and plot quantifications of voxel annotations such as drusen. Check out the [documentation](https://MedVisBonn.github.io/eyepy), especially the [Cookbook](https://medvisbonn.github.io/eyepy/cookbook/) chapter, for more information.

## Features

* Import Data (Heyex-E2E, Heyex-VOL, Heyex-XML, Topcon-FDA, B-Scan collections, [RETOUCH Challenge](https://retouch.grand-challenge.org/), [AMD Dataset from Duke University](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm))
* Analyze OCT volumes (compute and quantify drusen)
* Visualize OCT volumes with annotations and quantifications
* Save and load EyeVolume objects

## Getting Started

### Installation
**Attention:** If you want to use a version prior to 0.12.0 you have to install from the `eyepie` name instead. This is because we used `eyepie` as a package name on PyPI until the previous owner of the `eyepy` name on PyPI was so kind to transfer it to us.

To install the latest version of eyepy run `pip install -U eyepy`. (It is `eyepie` for versions < 0.12.0)

### Getting Started
When you don't hava a supported OCT volume at hand you can check out our sample dataset to get familiar with `eyepy`.

```python
from eyepy.data import load
ev = load("drusen_patient")
```

If you have data at hand use one of eyepys import functions.

```python
# Import HEYEX E2E export
ev = ep.import_heyex_e2e("path/to/file.e2e")
# Import HEYEX XML export
ev = ep.import_heyex_xml("path/to/folder")
# Import HEYEX VOL export
ev = ep.import_heyex_vol("path/to/file.vol")
# Import Topcon FDA export
ev = ep.import_topcon_fda("path/to/file.fda")
# Import volume from Duke public dataset
ev = ep.import_duke_mat("path/to/file.mat")
# Import volume form RETOUCH challenge
ev = ep.import_retouch("path/to/volume_folder")
```

# Related Projects:

+ [eyeseg](https://github.com/MedVisBonn/eyeseg): A python package for segmentation of retinal layers and drusen in OCT data.
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
