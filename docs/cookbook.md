# Cookbook

Here you learn how to use eyepy to perform common tasks.

## Import OCT data

Currently `eyepy` supports the HEYEX E2E, VOL and XML formats, as well as reading data from several public OCT datasets. It is recommended to use one of the following functions to import OCT data into `eyepy`:

``` python
import eyepy as ep

eye_volume = ep.import_bscan_folder("path/to/folder")
eye_volume = ep.import_duke_mat("path/to/file.mat")
eye_volume = ep.import_retouch("path/to/file.txt")
eye_volume = ep.import_heyex_xml("path/to/file.xml")
eye_volume = ep.import_heyex_vol("path/to/file.vol")
eye_volume = ep.import_heyex_e2e("path/to/file.e2e", single=True)
```

All functions return a single `EyeVolume` object representing the data. E2E files may contain several oct volumes which you can retrieve by setting the parameter `single` to `False`:

!!! Warning "Missing scale information"
    For the E2E file the scale of both localizer axes as well as the B-scan x-axes has not been identified yet and is hardcoded. When importing B-scans from folders there is also no scale information available.


## Plot Localizer

Most OCT volumes come with a localizer image. This image can be plotted using the `plot` method of the `EyeVolume` object:

``` python
eye_volume.plot()
```

There are several options to customize the plot:
    + show B-scan positions or region (`bscan_positions` and `bscan_region` parameters)
    + show enface projections or quantifications of OCT voxel annotaions such as drusen (`projections` and `quantifications` parameters)
    + show only a specific region of the localizer image (`region` parameter)

The plotting function is documented here: [EyeVolume.plot][eyepy.core.EyeVolume.plot]

!!! Warning `region` parameter
    The region parameter produces unexpected results in combination with `bscan_positions` and `bscan_region` parameters
## Plot Bscans

B-scans can be plotted using the `plot` method of the `EyeBscan` object. You get `EyeBscan` objects by indexing the `EyeVolume` object or iterating over it. The following code plots the first B-scan of the volume together with the layer annotations for BM and RPE:

``` python
eye_volume[0].plot(layers=["BM", "RPE"])
```

The plotting function is documented here: [EyeBscan.plot][eyepy.core.EyeBscan.plot]

<!---
## Access Meta data

## Modify Annotations

### Add / Remove Layer Annotations

### Add / Remove Voxel Annotaitons

### Add / Remove A-scan Annotations

### Add / Remove Shape Annotations

### Compute Drusen from Layer Annotations

## Quantify Annotations

### ETDRS and Custom Quantification Grids

### Map between Localizer and OCT space

## Registration of Enface Images
-->
