In contrast to the VOL and XML exports the E2E data may contain several OCT volumes and other data. The format might even allow for data from multiple patients in a single E2E file. Everything is stored in a general container structure that is described in the [File Structure](#file-structure) section.

# File Structure
The first bytes in an E2E file contain a version structure followed by a header structure. The header gives you access to the rest of the file by identifying the position of the last chunk of data. Each chunk has exactly 512 elements which we call folders.

{{ get_structure_doc("E2EFormat") }}

### Chunk
Every chunk has a header similar to the file header. A chunk then holds the headers of all contained folders sequentially, followed by data containers, that are referenced by the folder headers. A chunk can contain folders with data of different patients, studies, series, slices and types. Each folder contains data for a single (patient, study, series, slice, type) combination which is given in the folder header as well as the data container header. For the last chunk to have 512 folders, empty folders of `type=0` are appended.


{{ get_structure_doc("Chunk") }}


### Data Container

{{ get_structure_doc("DataContainer") }}

## Data Items

While the most important data, images and annotations were identified, there are still many data items that are not understood. We choose to sort the found data items by the level of information they are likely to contain. Therefore we use the IDs provided in the ContainerHeader (patient ID, study ID, series ID and slice ID). We assume that as in our test data these IDs follow some rules in them beeing hierarchical. Having a study is only meaningful if there is a patient and having a series in a study requires a study. Finally a slice requires a series to be contained in.

1. Hence, if a slice ID is given we assume that the data is slice specific.
2. If this is not the case, but a series ID is given, the data is series specific.
3. If this is not the case, but a study ID is given, the data is study specific.
4. If this is not the case, but a patient ID is given, the data is patient specific.
5. If no ID is given, the data is general or a filler.

In the following sections we describe the data items we found in each level of the described hierarchy. If you have any further information, please open an [issue](https://github.com/MedVisBonn/eyepy/issues) on GitHub and let us know.

### Slice Data

{{ get_hierarchy_doc("E2ESliceStructure") }}

### Series Data

{{ get_hierarchy_doc("E2ESeriesStructure") }}

### Study Data

{{ get_hierarchy_doc("E2EStudyStructure") }}

### Patient Data

{{ get_hierarchy_doc("E2EPatientStructure") }}

### General Data

{{ get_hierarchy_doc("E2EFileStructure") }}


## Further observations

+ Some type IDs for example the laterality type might be repeated. In our test data there is a case where the only difference between the folder headers of this laterality data is the value of the `unknown2` field which takes the values 0, 1 and 65535.
