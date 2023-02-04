# File Formats

## XML (Heidelberg)

## VOL (Heidelberg)

## E2E (Heidelberg)

###

The general structure of an E2E file looks like this:

+ version
+ header
+ List of chunks where each chunk:
    + has a header
    + has a list of folders where each folder:
        + has a header
        + has a data container with:
            + a header
            + an item

Each chunk contains exactly 512 folders. A chunk can contain folders with data of different patients, studies, series,slices and types but each folder contains data for a single (patient, study, series, slice, type) combination which is given in the folder header as well as the data container header.

The first chunk starts with 10 assumingly study specific folders (only patient and study are given), followed by assumingly Series specific folders (slice id is not given). Interestingly there are some type ids repeating here, including the laterality type. The only difference between the folder headers with the identical types is the value of the `unknown2` field which takes the values 0, 1 or 65535. The `unknown` field of the folder header is always 0. `unknown4` might be some kind of unique identifier since in the test data there are no duplicates while `unknown3` might be some kind of a foreign key.

Then the folders for every slice of all series follow. There might be multiple folders of the layer type for each slice, each holding a different layer for the respective B-scan. After the last slice 6 patient specific folders follow where only the patient_id is given and finally there is a folder of type 9011 which is the last folder in the last chunk before the filler folders start.

We are currently parsing only a small portion of the available folders, to extract image and meta data. We extract data of the following types and sizes:

Type: 9 - Size: 131   # Patient data
Type: 11 - Size: 27  # Laterality data
Type: 10004 - Size: 428  # Bscan meta data
Type: 10019 - Size: ~3108  # Layer data
Type: 1073741824 - Size: ~761876  # Image data

### The mysteries of the E2E format
There are several data folder types `HeE2eReader` currently does not support. If you believe the data you are looking for is in here, you can access the data using the hidden `_unknown_folders` attribute of the `HeE2eReader`. You will get a dictionary with keys in the format (PatientID, StudyID, SeriesID, SliceID, Type). The values are the parsed objects. You can access the unparsed binary data via the `.data_container.item` attribute. Very small entries might not have a `data_container` and can be `None` instead.

From here it is up to you to figure out the meaning of the data. I would appreciate if you share your findings.

Here is a list of all the data currently not understood by the `HeE2eReader`. Whenever one of the IDs is maxint the respective data has to be more general. The size given here might vary depending on the respective data.

**Study fields (Series is maxint):**

    Type: 9000 - Size: 264
    Type: 9001 - Size: 776
    Type: 58 - Size: 91
    Type: 7 - Size: 68
    Type: 1000 - Size: 51
    Type: 53 - Size: 51
    Type: 13 - Size: 200 (# Containtes OCT + HRA string)
    Type: 10 - Size: 91
    Type: 30 - Size: 2

**Series fields (Slice is -1)**

    Type: 9006 - Size: 520
    Type: 9005 - Size: 264
    Type: 9007 - Size: 520
    Type: 9008 - Size: 520
    Type: 59 - Size: 27
    Type: 1003 - Size: 17
    Type: 1000 - Size: 51
    Type: 62 - Size: 228
    Type: 10013 - Size: 2212
    Type: 10005 - Size: 24
    Type: 10009 - Size: 4
    Type: 10025 - Size: 100 # Slodata
    Type: 61 - Size: 4
    Type: 10010 - Size: 4112
    Type: 54 - Size: 97
    Type: 1001 - Size: 56
    Type: 3 - Size: 96
    Type: 2 - Size: 2274 (Preview? contains letters (JFIF))
    Type: 10011 - Size: 4
    Type: 1008 - Size: 8
    Type: 1073751824 - Size: 51220
    Type: 1073751825 - Size: 51220
    Type: 1073751826 - Size: 24596



**Patient fields (study and series are maxint)**

    Type: 52 - Size: 97
    Type: 9010 - Size: 269064
    Type: 31 - Size: 217
    Type: 17 - Size: 2
    Type: 29 - Size: 2

**General fields (patient study and series are maxint)**

    Type: 9011 - Size: 64664 # Last element in the last chunk before fillers fill the chunk
    Type: 0 - Size: 0 # Filler at the end of the parsed file the last chunk is filled with folders of this type until there are 512 folders in the chunk

**Bscan fields:**

    Type: 5 - Size: 59
    Type: 39 - Size: 497
    Type: 3 - Size: 96
    Type: 2 - Size: 2074 (Preview? contains letters (JFIF))
    Type: 10012 - Size: 100
    Type: 40 - Size: 28

## Topcon

## Zeiss
