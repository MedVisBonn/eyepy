<h1 align="center">eyepy</h1>
<p align="center">
A powerful Python package for importing, analyzing, and visualizing retinal imaging data, including OCT and OCT Angiography.
</p>

![header_gif](https://user-images.githubusercontent.com/5720058/228815448-4b561246-dac9-4f8f-abde-e0dd5457a72b.gif)

[![Documentation](https://img.shields.io/badge/docs-eyepy-blue)](https://MedVisBonn.github.io/eyepy)
[![PyPI version](https://badge.fury.io/py/eyepy.svg)](https://badge.fury.io/py/eyepy)
[![DOI](https://zenodo.org/badge/292547201.svg)](https://zenodo.org/badge/latestdoi/292547201)

`eyepy` provides a unified and user-friendly interface for working with retinal imaging data. With support for a wide range of file formats, it enables researchers and clinicians to import, process, and visualize OCT volumes and angiography data with ease. The core `EyeVolume` object offers intuitive methods for plotting fundus images, B-scans, and quantitative analyses such as drusen and retinal layer thickness. Comprehensive documentation and example workflows are available to help you get started quickly.

## Features

* Import Structural Data (HEYEX-E2E, HEYEX-VOL, HEYEX-XML, Topcon-FDA, B-Scan collections, [RETOUCH Challenge](https://retouch.grand-challenge.org/), [AMD Dataset from Duke University](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm))
* Import Angiographic OCT Data (HEYEX-VOL)
* Analyze OCT volumes (compute and quantify drusen)
* Visualize OCT volumes with annotations and quantifications
* Compute and visualize retinal layer thickness
* Compute and visualize OCTA enface projections.
* Save and load EyeVolume objects

## Getting Started

### Installation
**Attention:** If you want to use a version prior to 0.12.0 you have to install from the `eyepie` name instead. This is because we used `eyepie` as a package name on PyPI until the previous owner of the `eyepy` name on PyPI was so kind to transfer it to us.

To install the latest version of eyepy run `pip install -U eyepy`. (It is `eyepie` for versions < 0.12.0)

#### Optional Dependencies
Some file formats require additional dependencies:
- **Topcon FDA files**: `pip install eyepy[fda]` (requires `oct-converter`)
- **RETOUCH dataset**: `pip install eyepy[itk]` (requires `itk`)

### Getting Started
When you don't have a supported OCT volume at hand you can check out our sample datasets to get familiar with `eyepy`.

```python
from eyepy.data import load
struc_ev = load("drusen_patient")
struc_ev = load("healthy_OD")
angio_ev = load("healthy_OD_Angio")
```

If you have data at hand use one of eyepy's import functions.

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
# Import volume from RETOUCH challenge
ev = ep.import_retouch("path/to/volume_folder")
# Import HEYEX OCTA VOL export
ev_angio = ep.import_heyex_angio_vol("path/to/volume_folder")
```

## Spectralis OCTA (OCT Angiography) Support

`eyepy` is capable of reading and visualizing OCT Angiography (OCTA) data from Heidelberg Spectralis devices. You can explore and analyze both structural and angiography volumes using the same unified interface.

### Example: Load and Visualize Spectralis OCTA Sample Data

The following example demonstrates how to load OCTA sample data, and plot the enface projections.

```python
import eyepy as ep
import matplotlib.pyplot as plt

# Load sample data
angio_OD = ep.data.load("healthy_OD_Angio")
angio_OS = ep.data.load("healthy_OS_Angio")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for i, (angio, title) in enumerate(zip([angio_OD, angio_OS], ["Right Eye (OD)", "Left Eye (OS)"])):
    # Show localizer with Angiography overlay for the complete retina
    angio.plot(ax=axes[i], slabs=["RET"])
    axes[i].set_title(title)
    axes[i].axis("off")

plt.tight_layout()
```
![Example OCTA](https://github.com/user-attachments/assets/95b73e2b-0387-40cc-a09c-2765a8b2096a)


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


# Citation

If you use eyepy in your research, please cite it. You can find citation information and export BibTeX entries via the Zenodo record: [![DOI](https://zenodo.org/badge/292547201.svg)](https://zenodo.org/badge/latestdoi/292547201)

# Contributing

For details on contributing and setting up a development environment, see the [Contributing Guide](https://medvisbonn.github.io/eyepy/contributing/).
