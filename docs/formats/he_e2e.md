# Heidelberg Engineering E2E Format

Missing documentation of the Heidelberg E2E format has caused frustration by many working with OCT data and several projects tried to make the data accessible. Here you learn how to conveniently access data from E2E files using our [`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader] and what we know about the format.

## Get to know your data

One thing that makes it especially difficult to read data from E2E files is that E2E is a general container format that can contain 0 or more instances of different kinds of data. If you are interested how the data is stored you might want to continue [here](he_e2e_structures/he_e2e_structure_doc.md).

For most users it is probably enough to print the [`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader] object to get an overview of Patients, Studies and Series stored in the file.

```python title="Print E2E file overview"
from eyepy.io import HeE2eReader
print(HeE2eReader("filename.E2E")) # (1)
```

1.  Printing an HeE2eReader results in something similar to this:

    ```text
    E2EFile


        E2EPatient(321)


        E2EStudy(1234)
        Device: Heidelberg Retina Angiograph - Studyname: NAME

            E2ESeries(50001) - Laterality: OD - B-scans: 241
            Structure: Retina - Scanpattern: OCT ART Volume - Oct Modality: OCT - Enface Modality: IR

            E2ESeries(50002) - Laterality: OD - B-scans: 25
            Structure: Retina - Scanpattern: OCT ART Volume - Oct Modality: OCT - Enface Modality: IR
    ```

## Access data that can be parsed to [`EyeVolume`][eyepy.core.EyeVolume] objects

The [`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader] provides a convenient interface to access data stored in E2E files. Assuming that your E2E file contains one or more OCT volumes you can parse the volumes to [`EyeVolume`][eyepy.core.EyeVolume] objects using the following code:

```python
from eyepy.io import HeE2eReader

with HeE2eReader("filename.E2E") as e2e_reader: # (1)
    volumes = e2e_reader.volume

with HeE2eReader("filename.E2E", single=True) as e2e_reader: # (2)
    volume = e2e_reader.volume
```

1.  `e2e_reader.volume` returns `List[EyeVolume]`
2.  `e2e_reader.volume` returns the first Series as `EyeVolume` object

!!! Warning Limitations of EyeVolume objects created from E2E files
    + Currently we can not read scale Information for the localizer images as well as the x-scale of the B-scans. Hence quantifications can not be transformed to metric units.
    + Also we know that in the E2E file B-scans are not registered with each other. This B-scan registration information has also not been found yet. This might cause problems when downstream analysis expects B-scans to be registered.

    If you know how to read this information from the E2E file, please let us know by opening an [issue](https://github.com/MedVisBonn/eyepy/issues)

## Access other data stored in E2E files

Not everything stored in an E2E file is accessible through parsing to [`EyeVolume`][eyepy.core.EyeVolume] objects. If you are interested in accessing other data stored in the E2E file, you can use the file hierarchy created by the [`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader]. The structure of the build hierarchy is shown in the [diagram](#e2e-hierarchy) below. The file hierarchy can be accessed through the `file_hierarchy` attribute of the [`HeE2eReader`][eyepy.io.he.e2e_reader.HeE2eReader] object. You can either traverse the hierarchy level by level or access all elements of a specific level at once using one of the following attributes:

+ `e2e_reader.patients` returns a list of all [`E2EPatientStructure`][eyepy.io.he.e2e_reader.E2EPatientStructure] objects
+ `e2e_reader.studies` returns a list of all [`E2EStudyStructure`][eyepy.io.he.e2e_reader.E2EStudyStructure] objects
+ `e2e_reader.series` returns a list of all [`E2ESeriesStructure`][eyepy.io.he.e2e_reader.E2ESeriesStructure] objects

First you might want to get an overview about the data stored in the hierarchy. Therefore you can use the following code:

```python
from eyepy.io import HeE2eReader

with HeE2eReader("filename.E2E") as e2e_reader:
    print(e2e_reader.inspect(recursive=True)) # (1)
```

1.  This method is basically an extended version of `print(HeE2eReader("filename.E2E"))` that adds for every level of the hierarchy a table with information about the containded data.


### E2E Hierarchy
```mermaid
classDiagram
    E2EFileStructure *-- E2EPatientStructure
    E2EPatientStructure *-- E2EStudyStructure
    E2EStudyStructure *-- E2ESeriesStructure
    E2ESeriesStructure *-- E2ESliceStructure

    E2EStructureMixin <|-- E2EFileStructure
    E2EStructureMixin <|-- E2EPatientStructure
    E2EStructureMixin <|-- E2EStudyStructure
    E2EStructureMixin <|-- E2ESeriesStructure
    E2EStructureMixin <|-- E2ESliceStructure

    class E2EStructureMixin{
      - inspect(recursive, ind_prefix, tables)
      - get_folder_data(folder_type, offset, data_construct)
    }

    class E2EFileStructure{
      - folders: Dict[Union[TypesEnum, int], E2EFolder]
      - substructure/patients: Dict[int, E2EPatientStructure]
    }
    class E2EPatientStructure{
      - id: int
      - folders: Dict[Union[TypesEnum, int], E2EFolder]
      - substructure/studies: Dict[int, E2EStudyStructure]
    }
    class E2EStudyStructure{
      - id: int
      - folders: Dict[Union[TypesEnum, int], E2EFolder]
      - substructure/series: Dict[int, E2ESeriesStructure]
    }
    class E2ESeriesStructure{
      - id: int
      - folders: Dict[Union[TypesEnum, int], E2EFolder]
      - substructure/slices: Dict[int, E2ESlice]
      - get_volume() -> EyeVolume
      - get_layers() -> Dict[int, np.ndarray]
      - get_localizer() -> EyeEnface
      - get_localizer_meta() -> EyeEnfaceMeta
      - get_meta() -> EyeVolumeMeta
      - get_bscan_meta() -> List[EyeBscanMeta]
    }
    class E2ESliceStructure{
      - id: int
      - folders: Dict[Union[TypesEnum, int], E2EFolder]
      - get_layers() -> Dict[int, np.ndarray]
      - get_image() -> -> np.ndarray
      - get_meta() -> EyeBscanMeta
    }
```



If you have any further information on the E2E format or if you find any errors in this document, please let us know by opening an [issue](https://github.com/MedVisBonn/eyepy/issues).

!!! Warning  "Open questions and differences to other Heidelberg Formats"
    + B-scan positions in the E2E format are given relative to an origin roughly in the center of the localizer image. We assume that the positions are given as angles in degree since the absolute value of minimum and maximum position is very close to half the field of view. This is different to VOL and XML formats where positions are given in mm with the origin in the top left corner of the localizer image. Since some position values indicate that they are located outside of the localizer image, we might have to apply the localizer transformation to them as well after mapping them to pixel indices.
    + VOL and XML exports store the localizer scaling, as well as the scaling of the B-scans. The VOL format even stores the distance between the B-scans which has to be calculated from the B-scans in the XML and currently also the E2E format. We did not find this scaling information in the E2E format yet and use a hardcoded value for now. The only scaling we found was the Y Scale of the B-scan.

## HEYEX Metadata Provenance

The [`E2ESeriesStructure`][eyepy.io.he.e2e_reader.E2ESeriesStructure] now exposes a HEYEX-style metadata view:

```python
from eyepy.io import HeE2eReader

with HeE2eReader("filename.E2E") as reader:
    series = reader.series[0]
    series_metadata = series.get_heyex_metadata()
    bscan_metadata = series.get_heyex_metadata(bscan_index=0)
    sources = series.get_heyex_metadata_sources(bscan_index=0)
    volume = series.get_volume()
    volume_bundle = volume.meta["e2e_metadata"]
```

The `bscan_index` argument matters because several OCT export fields are slice specific in HEYEX. If `bscan_index` is omitted, the reader leaves `OCT Image -> ART Mode`, `Quality`, and `Acquisition Time` as `None`.

When an E2E series is converted to an [`EyeVolume`][eyepy.core.EyeVolume], the same information is also attached under `volume.meta["e2e_metadata"]`. This bundle contains the series-level HEYEX metadata, provenance, per-B-scan HEYEX OCT metadata, and the current provisional findings.

### Field Scope

| Scope | Meaning |
| --- | --- |
| `patient` | Read from patient-level containers such as `Type9` |
| `study` | Read from study-level containers such as `Type13`, `Type9000`, `Type9001` |
| `series` | Read from series-level containers such as `Type10005`, `Type9005`-`Type9008` |
| `bscan` | Read from slice/B-scan containers such as `Type39` or `Type10004` |
| `derived` | Computed directly from stored values |
| `provisional` | Strong empirical mapping, but not yet proven |

### Provenance Table

| Export field | Scope | Provenance |
| --- | --- | --- |
| `General Parameters -> Examination Time` | `series` | `Type10005@series+offset16` |
| `General Parameters -> Examined Structure` | `series` | `Type9005` |
| `General Parameters -> Application` | `study/series` | `Type13`, with fallback to `Type9005` |
| `General Parameters -> Scan Focus` | `bscan/provisional` | `Type10004@offset140` as `float32`, converted empirically |
| `IR Image -> ART Mode` | `bscan` | `Type39@offset376` |
| `IR Image -> Sensitivity (DC/DC)` | `bscan` | `Type39@offset8` |
| `IR Image -> Total Sensitivity` | `bscan` | `Type39@offset116` |
| `OCT Image -> ART Mode` | `bscan` | `Type10004@offset120` |
| `OCT Image -> Quality` | `bscan` | `Type10004@offset156` |
| `OCT Image -> Acquisition Time` | `bscan` | `Type10004.acquisitionTime` |
| `OCT Image -> Scaling Z` | `bscan` | `Type10004.scale_y` |
| `OCT Image -> Scaling X` | `derived/provisional` | `Type10004.start_x`, `end_x`, and `imgSizeWidth` |
| `OCT Scan Pattern -> Number of B-Scans` | `bscan` | `Type10004.n_bscans` |
| `OCT Scan Pattern -> Pattern Size` | `derived/provisional` | angular span from `Type10004`, converted with the current `Medium` eye-length constant |
| `OCT Scan Pattern -> Distance between B-Scans` | `derived/provisional` | vertical span from `Type10004`, converted with the current `Medium` eye-length constant |
| `Device -> Camera Model` | `study` | `Type13` |
| `Device -> Camera Model Code` | `bscan` | ASCII model codes embedded in `Type39` |
| `Device -> Camera/Power/Touch serial numbers` | `bscan` | `Type39@offset104`, `108`, `112` |
| `Device -> HRA/Power/Touch firmware` | `bscan` | `Type39@offset128`, `132`, `136` |
| `Device -> OCT Controller FW Version` | `bscan` | `Type10004@offset124` |
| `Device -> OCT Camera FW Version` | `bscan/provisional` | `Type10004@offset128` |
| `Device -> OCT Camera FPGA Version` | `bscan/provisional` | `Type10004@offset132` |
| `Device -> Acquisition Software Version` | `bscan` | `Type39@offset140` |

### Provisional Mappings

`Scan Focus` currently uses the strongest candidate field found:

```text
v = Type10004@offset140 as float32
focus_D ≈ 1.079 * sign(v) * (abs(v) - 3.505)
```

This fit matches our examples closely, but it is still empirical and should be treated as provisional.

`Scaling X` in `µm/pixel` also appears to be derived rather than stored directly:

```text
Scaling X ≈ (end_x - start_x) / imgSizeWidth * 289.6
```

The constant `289.6 µm/degree` is the current best fit for the `Medium` eye-length preset. Additional paired files are still needed to confirm the conversion for `Short` and `Long`.

### Current Caveats

+ `Type40` currently looks related to acquisition rotation / shear geometry rather than focus. In our sample set it changes for the rotated scan but not for the other focus-varying scans.
+ `OCT Camera FW Version` and `OCT Camera FPGA Version` are assigned from `Type10004@offset128` and `Type10004@offset132` respectively, but these two slots may need to be swapped later if a dataset with distinct values proves the order is reversed.
+ `Date of Birth`, `Resolution Mode`, `Camera Objective`, `Internal Target`, `External Target`, `A-Scan Rate`, `Eye Length`, `EDI Mode`, and `EVI Mode` are still unresolved and are therefore returned as `None` by the HEYEX metadata helper.

## Acknowledgements
While building the E2E file reader, and investigating the format we took inspiration from several existing projects, which we would like to thank:

+ [OCT-Converter](https://github.com/marksgraham/OCT-Converter)
+ [LibE2E](https://github.com/neurodial/LibE2E)
+ [uocte](https://bitbucket.org/uocte/uocte/wiki/Home)
+ [RETIMAT](https://github.com/drombas/retimat)
