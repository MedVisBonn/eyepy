# Cookbook

Here you learn how to use eyepy to perform common tasks.

## Import OCT data

Currently `eyepy` supports the HEYEX E2E, VOL and XML formats, as well as reading data from several public OCT datasets. All functions return a single `EyeVolume` object representing the data. E2E files may contain several OCT volumes which you can retrieve by setting the parameter `single` to `False`. While you can use Reader objects to parse the data and access specific information, it is recommended to use the provided import functions to get `EyeVolume` object which are a convenient interface to the data that provides a unified interface to data imported from various sources.

```python
import eyepy as ep
# Import HEYEX E2E export
ev = ep.import_heyex_e2e("path/to/file.e2e", single=True)
# Import HEYEX XML export
ev = ep.import_heyex_xml("path/to/folder")
# Import HEYEX VOL export
ev = ep.import_heyex_vol("path/to/file.vol")
```

!!! Warning "Missing scale information"
    For the E2E file the scale of both localizer axes as well as the B-scan x-axes has not been identified yet and is hardcoded. When importing B-scans from folders there is also no scale information available.

When only B-scans exist in a folder `eyepy` might still be able to import them. B-scans are expected to be ordered and distributed with equal distance on a quadratic area.

```python
import eyepy as ep
# Import B-scans from folder
ev = ep.import_bscan_folder("path/to/folder")
```

Public OCT datasets often have their own data formats. `eyepy` can import volumes from the [AMD Dataset from Duke University](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) and the [RETOUCH Challenge](https://retouch.grand-challenge.org/).

```python
import eyepy as ep
# Import DUKE volume
ev = ep.import_duke_mat("path/to/volume.mat")
# Import RETOUCH volume
ev = ep.import_retouch("path/to/folder")
```

When you don't hava a supported OCT volume at hand you can check out our sample dataset to get familiar with `eyepy`.
```python
from eyepy.data import load
# Import HEYEX XML export
ev = load("drusen_patient")
```

## Plot Localizer

Most OCT volumes come with a localizer image. This image can be plotted using the `plot` method of the `EyeVolume` object:

```python
ev.plot()
```

There are several options to customize the plot:
    + show B-scan positions or region (`bscan_positions` and `bscan_region` parameters)
    + show enface projections or quantifications of OCT voxel annotaions such as drusen (`projections` and `quantifications` parameters)
    + show only a specific region of the localizer image (`region` parameter)

The plotting function is documented here: [EyeVolume.plot][eyepy.core.EyeVolume.plot]

!!! Warning "`region` parameter"
    The region parameter produces unexpected results in combination with `bscan_positions` and `bscan_region` parameters

## Plot Bscans

B-scans can be plotted using the `plot` method of the `EyeBscan` object. You get `EyeBscan` objects by indexing the `EyeVolume` object or iterating over it. The following code plots the first B-scan of the volume together with the layer annotations for BM and RPE:

``` python
ev[0].plot(layers=["BM", "RPE"])
```

The plotting function is documented here: [EyeBscan.plot][eyepy.core.EyeBscan.plot]

## Modify Annotations

### Compute Drusen from Layer Annotations

Here we compute drusen for our sample data which has manual layer annotations for BM and RPE.

``` python
import eyepy as ep

# Import example data
ev = ep.data.load("drusen_patient")
# Compute drusen
drusen_map = ep.drusen(ev.layers["RPE"].data, ev.layers["BM"].data, ev.shape, minimum_height=2)
```

### Add / Remove Layer Annotations
Often OCT volumes come with layer annotations. They are added to the `EyeVolume` object during the import, but you can also manipulate them yourself using the `add_layer_annotation` and `remove_layer_annotation` methods. The following code can be used to layer annotations to `EyeVolume` objects. The `name` parameter is used to identify the layer.

```python
layer_heights = np.zeros(np.array(ev.shape)[[0,2]])
ev.add_layer_annotation(layer_heights, name="new_layer")
```

To remove a layer annotation use the `remove_layer_annotation` method. The following code removes the layer annotation we added above.

```python
ev.remove_layer_annotation("new_layer")
```

### Add / Remove Voxel Annotaitons
If you want to add voxel annotations to the EyeVolume object you can use the `add_voxel_annotation` method. The following code adds the drusen map we computed above to the EyeVolume object. The `name` parameter is used to identify the annotation.

```python
ev.add_voxel_annotation(drusen_map, name="drusen")
```

To remove an annotation use the `remove_voxel_annotation` method. The following code removes the drusen annotation we added above.

```python
ev.remove_voxel_annotation("drusen")
```

### ETDRS and Custom Quantification Grids

Quantifications on circular grids such as the ETDRS grid are common to quantify imaging data of the eye. With `eyepy` you can easily compute quantifications on such grids. The following code computes a quantification grid for the drusen annotation we added above.

```python
fig, axes = plt.subplots(1, 2, figsize=(5, 10))

# Configure quantification grid for drusen quantification
ev.volume_maps["drusen"].radii = [1.5, 2.5]
ev.volume_maps["drusen"].n_sectors = [4, 8]
ev.volume_maps["drusen"].offsets = [0, 45]

# Plot drusen projection and quantification
ev.plot(ax=axes[0], projections=["drusen"])
ev.plot(ax=axes[1], quantification="drusen")
axes[0].axis("off")
axes[1].axis("off")
```

The result looks like this: On the left, the scale is the drusen height in voxel and on the right, the drusen volume in mm³

![Example quantification](https://user-images.githubusercontent.com/5720058/218107881-841c224a-ca1c-465f-ab42-7aa3726fb991.jpeg)

To access the quantification as a dictionary use `ev.volume_maps["drusen"].quantification`

### Interact with individual B-scans
If you index into an EyeVolume object you get EyeBscan objects. Annotations you added to the respective `EyeVolume` object are also available in the `EyeBscan` object and can be visualized easily. The following code plots the 40th B-scan of the volume together with the layer annotations for BM and RPE and the computed drusen annotation:

```python
import numpy as np

fig, ax = plt.subplots(1,1, figsize=(9,3))
bscan = ev[40]
bscan.plot(layers=["BM", "RPE"], areas=["drusen"], region=np.s_[150:300, :], ax=ax)
ax.axis("off")
```

![B-scan visualization](https://user-images.githubusercontent.com/5720058/218107633-fdb51f92-7415-4673-aef5-f8cbedda970e.jpeg)

<!---
## Access Meta data

## Modify Annotations



### Add / Remove A-scan Annotations

### Add / Remove Shape Annotations

## Quantify Annotations



### Map between Localizer and OCT space

## Registration of Enface Images
-->
