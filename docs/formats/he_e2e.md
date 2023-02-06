# Heidelberg Engineering E2E Format

Missing documentation of the Heidelberg E2E format has caused frustration by many working with OCT data and several projects tried to make the data accessible. While our understanding is still by no means complete, this document aims to provide a comprehensive description of what we know and what our E2E file reader (`HeE2eReader`) builds on. If you have any questions or suggestions, please open an issue on GitHub.


## File Structure
An E2E file starts with a version structure, followed by a header structure. After that the data comes in chunks of 512 folders.


| Structure   | Size (bytes) | Comment                                                                            |
| :---------- | -----------: | :--------------------------------------------------------------------------------- |
| `Version`   |           36 |                                                                                    |
| `Header`    |           52 | The `prev` field in the main header refers to the start position of the last chunk |
| `ChunkList` |       varies | The number and size of the chunks depends on the data                              |

### Chunk
Every chunk has a header similar to the file header. A chunk then holds the headers of all contained folders sequentially, followed by data containers, that are referenced by the folder headers. A chunk can contain folders with data of different patients, studies, series, slices and types. Each folder contains data for a single (patient, study, series, slice, type) combination which is given in the folder header as well as the data container header. For the last chunk to have 512 folders, empty folders of `type=0` are appended.

| Structure        | Size (bytes) | Comment                                                                      |
| :--------------- | -----------: | ---------------------------------------------------------------------------- |
| ` ChunkHeader`   |           52 | Each chunk refers to the start position of the previous chunk (`prev` field) |
| `FolderHeaders`  | 44*512=22528 | Each chunk has 512 folders with headers of size 44                           |
| `DataContainers` |       varies |                                                                              |

### Data Container
| Structure         | Size (bytes) | Comment                                                                                          |
| ----------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `ContainerHeader` | 60           |                                                                                                  |
| `DataItem`        | varies       | There are many kinds of DataItems indicated by different type IDs in the folder/container header |


## Data Items

While the most important data, images and annotations were identified, there are still many data items that are not understood. We choose to sort the found data items by the level of information they are likely to contain. Therefore we use the IDs provided in the ContainerHeader (patient ID, study ID, series ID and slice ID). We assume that as in our test data these IDs follow some rules in them beeing hierarchical. Having a study is only meaningful if there is a patient and having a series in a study requires a study. Finally a slice requires a series to be contained in.

1. Hence, if a slice ID is given we assume that the data is slice specific.
2. If this is not the case, but a series ID is given, the data is series specific.
3. If this is not the case, but a study ID is given, the data is study specific.
4. If this is not the case, but a patient ID is given, the data is patient specific.
5. If no ID is given, the data is general or a filler.

In the following sections we describe the data items we found. If you have any further information, please open an issue on GitHub and let us know.

### Slice Data

|    type ID |    size | name      | notes                            |
| ---------: | ------: | :-------- | :------------------------------- |
|          2 | 2377.62 | unknown2  | Preview? contains letters (JFIF) |
|          3 |      96 |           |                                  |
|          5 |      59 |           |                                  |
|         39 |     497 |           |                                  |
|         40 |      28 |           |                                  |
|      10004 |     428 | bscanmeta |                                  |
|      10012 |     100 |           |                                  |
|      10019 | 2825.84 | layer     | 1 folder per layer per slice     |
|      10032 |      92 |           |                                  |
| 1073741824 |  738008 | image     |                                  |


### Series Data

|    type ID |    size | name               | notes                            |
| ---------: | ------: | :----------------- | :------------------------------- |
|          2 | 2509.38 | unknown2           | Preview? contains letters (JFIF) |
|          3 |      96 |                    |                                  |
|         11 |      27 | laterality         |                                  |
|         54 |      97 |                    |                                  |
|         59 |      27 |                    |                                  |
|         61 |       4 |                    |                                  |
|         62 |     228 |                    |                                  |
|       1000 |      51 |                    |                                  |
|       1001 |   54.75 |                    |                                  |
|       1003 |      17 |                    |                                  |
|       1008 |       8 |                    |                                  |
|       9005 |     264 | examined_structure |                                  |
|       9006 |     520 | scanpattern        |                                  |
|       9007 |     520 | enface_modality    |                                  |
|       9008 |     520 | oct_modality       |                                  |
|      10005 |      24 |                    |                                  |
|      10009 |       4 |                    |                                  |
|      10010 |    4112 |                    |                                  |
|      10011 |       4 |                    |                                  |
|      10013 |   11284 |                    |                                  |
|      10025 |     100 | slodata            |                                  |
| 1073741824 |  589844 | image              |                                  |
| 1073751824 |   51220 |                    |                                  |
| 1073751825 |   51220 |                    |                                  |
| 1073751826 |   24596 |                    |                                  |

### Study Data

| type ID | size | name         | notes                     |
| ------: | ---: | :----------- | ------------------------- |
|       7 |   68 | measurements |                           |
|      10 |   91 |              |                           |
|      13 |  200 | unknown      | Contains OCT + HRA string |
|      30 |    2 | unknown5     |                           |
|      53 |   51 |              |                           |
|      58 |   91 |              |                           |
|    1000 |   51 |              |                           |
|    9000 |  264 | studyname    |                           |
|    9001 |  776 | device       |                           |

### Patient Data

| type ID |   size | name     | notes |
| ------: | -----: | :------- | :---- |
|       9 |    131 | patient  |       |
|      17 |      2 | diagnose |       |
|      29 |      2 | unknown4 |       |
|      31 |    217 |          |       |
|      52 |     97 |          |       |
|    9010 | 269064 | unknown1 |       |

### General Data

| type ID |  size | name         | notes                                         |
| ------: | ----: | :----------- | :-------------------------------------------- |
|       0 |     0 | empty_folder | Filler at the end of the last chunk           |
|    9011 | 64664 |              | Last element in the last chunk before fillers |


## Further observations

+ Folders in the chunks are grouped. The first chunk starts with 10 study specific folders followed by series specific folders, then the folders for all slices of all series follow. Finally, after the last slice, 6 patient specific folder follow. The last folder in the last chunk is of type 9011. After that there are only filler folders. Whether such a structure exists for every E2E file and could be used for faster access of specific data is not known.
+ Some type IDs for example the laterality type is repeated. The only difference between the folder headers is the value of the `unknown2` field which takes the values 0, 1 and 65535.

##
There are several data folder types `HeE2eReader` currently does not support.
If you believe the data you are looking for is in here, you can access the data using the hidden `_unknown_folders` attribute of the `HeE2eReader`. You will get a dictionary with keys in the format (PatientID, StudyID, SeriesID, SliceID, Type). The values are the parsed objects. You can access the unparsed binary data via the `.data_container.item` attribute.

From here it is up to you to figure out the meaning of the data. I would appreciate if you share your findings.


## Aknowledgements
While building the E2E file reader, and investigating the format we took inspiration from several exising projects, which we would like to thank:

+ [OCT-Converter](https://github.com/marksgraham/OCT-Converter)
+ [LibE2E](https://github.com/neurodial/LibE2E)
+ [uocte](https://bitbucket.org/uocte/uocte/wiki/Home)
+ [RETIMAT](https://github.com/drombas/retimat)


??? Version
    ```mermaid
    classDiagram
    class Version{
    name(12) ascii
    version(4) float32
    unknown(20)
    }
    ```
