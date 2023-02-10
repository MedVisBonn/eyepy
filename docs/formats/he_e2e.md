# Heidelberg Engineering E2E Format

Missing documentation of the Heidelberg E2E format has caused frustration by many working with OCT data and several projects tried to make the data accessible. While our understanding is still by no means complete, this document aims to provide a comprehensive description of what we know and what our E2E file reader (`HeE2eReader`) builds on.

In contrast to the VOL and XML exports the E2E data may contain several oct volumes. While our test data containes only data from a single patient it might be possible to have data from multiple patients in a single E2E file. After the patient, the next level of organization is the study. It seems that within a study all data for a single eye is bundled, which might be multiple OCT volumes and possibly other data. A series is a single OCT volume with localizer but might also be another modality. Finally, a slice is a single B-Scan. At least this is how we interpreted the structure of the data.

If you have any further information on the E2E format or if you find any errors in this document, please let us know by opening an [issue](https://github.com/MedVisBonn/eyepy/issues).
# File Structure
An E2E file starts with a version structure, followed by a header structure. After that the data comes in chunks of 512 folders.


| Structure                                 | Size (bytes) | Comment                                                                            |
| :---------------------------------------- | -----------: | :--------------------------------------------------------------------------------- |
| [`Version`](he_e2e_structures/version.md) |           36 |                                                                                    |
| [`Header`](he_e2e_structures/header.md)   |           52 | The `prev` field in the main header refers to the start position of the last chunk |
| `ChunkList`                               |       varies | The number and size of the chunks depends on the data                              |

### Chunk
Every chunk has a header similar to the file header. A chunk then holds the headers of all contained folders sequentially, followed by data containers, that are referenced by the folder headers. A chunk can contain folders with data of different patients, studies, series, slices and types. Each folder contains data for a single (patient, study, series, slice, type) combination which is given in the folder header as well as the data container header. For the last chunk to have 512 folders, empty folders of `type=0` are appended.

| Structure                                             | Size (bytes) | Comment                                                                      |
| :---------------------------------------------------- | -----------: | ---------------------------------------------------------------------------- |
| [`ChunkHeader`](he_e2e_structures/header.md)          |           52 | Each chunk refers to the start position of the previous chunk (`prev` field) |
| [`FolderHeaders`](he_e2e_structures/folder_header.md) | 44*512=22528 | Each chunk has 512 folders with headers of size 44                           |
| `DataContainers`                                      |       varies |                                                                              |

### Data Container
| Structure                                                  | Size (bytes) | Comment                                                                                          |
| ---------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| [`ContainerHeader`](he_e2e_structures/container_header.md) | 60           |                                                                                                  |
| `DataItem`                                                 | varies       | There are many kinds of DataItems indicated by different type IDs in the folder/container header |


## Data Items

While the most important data, images and annotations were identified, there are still many data items that are not understood. We choose to sort the found data items by the level of information they are likely to contain. Therefore we use the IDs provided in the ContainerHeader (patient ID, study ID, series ID and slice ID). We assume that as in our test data these IDs follow some rules in them beeing hierarchical. Having a study is only meaningful if there is a patient and having a series in a study requires a study. Finally a slice requires a series to be contained in.

1. Hence, if a slice ID is given we assume that the data is slice specific.
2. If this is not the case, but a series ID is given, the data is series specific.
3. If this is not the case, but a study ID is given, the data is study specific.
4. If this is not the case, but a patient ID is given, the data is patient specific.
5. If no ID is given, the data is general or a filler.

In the following sections we describe the data items we found. If you have any further information, please open an [issue](https://github.com/MedVisBonn/eyepy/issues) on GitHub and let us know.

### Slice Data

|    type ID |    size | name                                                      | notes                            |
| ---------: | ------: | :-------------------------------------------------------- | :------------------------------- |
|          2 | 2377.62 |                                                           | Preview? contains letters (JFIF) |
|          3 |      96 |                                                           |                                  |
|          5 |      59 |                                                           |                                  |
|         39 |     497 |                                                           |                                  |
|         40 |      28 |                                                           |                                  |
|      10004 |     428 | [bscanmeta](he_e2e_structures/bscanmeta.md)               |                                  |
|      10012 |     100 |                                                           |                                  |
|      10019 | 2825.84 | [layer_annotation](he_e2e_structures/layer_annotation.md) | 1 folder per layer per slice     |
|      10032 |      92 |                                                           |                                  |
| 1073741824 |  738008 | [image](he_e2e_structures/image.md)                       |                                  |


### Series Data

|    type ID |    size | name                                                | notes                            |
| ---------: | ------: | :-------------------------------------------------- | :------------------------------- |
|          2 | 2509.38 |                                                     | Preview? contains letters (JFIF) |
|          3 |      96 |                                                     |                                  |
|         11 |      27 | [laterality](he_e2e_structures/laterality.md)       |                                  |
|         54 |      97 |                                                     |                                  |
|         59 |      27 |                                                     |                                  |
|         61 |       4 |                                                     |                                  |
|         62 |     228 |                                                     |                                  |
|       1000 |      51 |                                                     |                                  |
|       1001 |   54.75 |                                                     |                                  |
|       1003 |      17 |                                                     |                                  |
|       1008 |       8 |                                                     |                                  |
|       9005 |     264 | [examined_structure](he_e2e_structures/textdata.md) |                                  |
|       9006 |     520 | [scanpattern](he_e2e_structures/textdata.md)        |                                  |
|       9007 |     520 | [enface_modality](he_e2e_structures/textdata.md)    |                                  |
|       9008 |     520 | [oct_modality](he_e2e_structures/textdata.md)       |                                  |
|      10005 |      24 |                                                     |                                  |
|      10009 |       4 |                                                     |                                  |
|      10010 |    4112 |                                                     |                                  |
|      10011 |       4 |                                                     |                                  |
|      10013 |   11284 |                                                     |                                  |
|      10025 |     100 | [slodata](he_e2e_structures/slodata.md)             |                                  |
| 1073741824 |  589844 | [image](he_e2e_structures/image.md)                 |                                  |
| 1073751824 |   51220 |                                                     |                                  |
| 1073751825 |   51220 |                                                     |                                  |
| 1073751826 |   24596 |                                                     |                                  |

### Study Data

| type ID | size | name                                              | notes                     |
| ------: | ---: | :------------------------------------------------ | ------------------------- |
|       7 |   68 | [measurements](he_e2e_structures/measurements.md) |                           |
|      10 |   91 |                                                   |                           |
|      13 |  200 |                                                   | Contains OCT + HRA string |
|      30 |    2 |                                                   |                           |
|      53 |   51 |                                                   |                           |
|      58 |   91 |                                                   |                           |
|    1000 |   51 |                                                   |                           |
|    9000 |  264 | [studyname](he_e2e_structures/textdata.md)        |                           |
|    9001 |  776 | [device](he_e2e_structures/textdata.md)           |                           |

### Patient Data

| type ID |   size | name                                      | notes |
| ------: | -----: | :---------------------------------------- | :---- |
|       9 |    131 | [patient](he_e2e_structures/patient.md)   |       |
|      17 |      2 | [diagnose](he_e2e_structures/textdata.md) |       |
|      29 |      2 |                                           |       |
|      31 |    217 |                                           |       |
|      52 |     97 |                                           |       |
|    9010 | 269064 |                                           |       |

### General Data

| type ID |  size | name         | notes                                         |
| ------: | ----: | :----------- | :-------------------------------------------- |
|       0 |     0 | empty_folder | Filler at the end of the last chunk           |
|    9011 | 64664 |              | Last element in the last chunk before fillers |


## Further observations

+ Folders in the chunks of our test data are grouped. The first chunk starts with 10 study specific folders followed by series specific folders, then the folders for all slices of all series follow. Finally, after the last slice, 6 patient specific folder follow. The last folder in the last chunk is of type 9011. After that there are only filler folders. Whether such a structure exists for every E2E file and could be used for faster access of specific data is not known.
+ Some type IDs for example the laterality type is repeated. The only difference between the folder headers is the value of the `unknown2` field which takes the values 0, 1 and 65535.

## Open questions and differences to other Heidelberg Formats

+ B-scans in the E2E format are not registered with each other. This might cause problems when downstream analysis expect them to be registered. We do not know whether registration information is stored in the E2E format but also did not search for it yet.
+ B-scan positions in the E2E format are given relative to an origin roughly in the center of the localizer image with an unknown scale.  This is different to VOL and XML formats where positions are given in mm with the origin in the top left corner of the localizer image. At the moment our HeE2eReader uses a hardcoded shift and scale that worked for our test data. It would be preferable to have a more general solution using data from the E2E file.
+ When comparing the localizer image retrieved from E2E and XML exports, we noticed that also the localizer might undergo some transformation, like shift and rotation which is not reflected in the E2E data. There might be an additional affine matrix stored in the E2E file that we did not find yet.
+ VOL and XML exports store the localizer scaling, as well as the scaling of the B-scans. The VOL format even stores the distance between the B-scans which has to be calculated from the B-scans in the XML and currently also the E2E format. We did not find this scaling information in the E2E format yet and use a hardcoded value for now. The only scaling we found was the Y Scale of the B-scan.

## Aknowledgements
While building the E2E file reader, and investigating the format we took inspiration from several existing projects, which we would like to thank:

+ [OCT-Converter](https://github.com/marksgraham/OCT-Converter)
+ [LibE2E](https://github.com/neurodial/LibE2E)
+ [uocte](https://bitbucket.org/uocte/uocte/wiki/Home)
+ [RETIMAT](https://github.com/drombas/retimat)
