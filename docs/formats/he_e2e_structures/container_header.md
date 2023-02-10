# Container Header Structure (60 bytes)

| name       | size | type   | description                                                                                                                                                                                 |
| ---------- | ---- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| magic3     | 12   | ascii  | Name of the version                                                                                                                                                                         |
| unknown0   | 4    | int32u |                                                                                                                                                                                             |
| header_pos | 4    | int32u | Position of the header                                                                                                                                                                      |
| pos        | 4    | int32u | Position of the data                                                                                                                                                                        |
| size       | 4    | int32u | Size of the data                                                                                                                                                                            |
| unknown1   | 4    | int32u | Always 0 (b'\x00\x00\x00\x00')? At leat in our data                                                                                                                                         |
| patient_id | 4    | int32s | Patient ID                                                                                                                                                                                  |
| study_id   | 4    | int32s | Study ID                                                                                                                                                                                    |
| series_id  | 4    | int32s | Series ID                                                                                                                                                                                   |
| slice_id   | 4    | int32s | Slice ID                                                                                                                                                                                    |
| ind        | 2    | int16u | 0 for enface and 1 for bscan for image containers                                                                                                                                           |
| unknown2   | 2    | int16u | Always 0 (b'\x00\x00')? At leat in our data                                                                                                                                                 |
| type       | 4    | int32u | Type of the data                                                                                                                                                                            |
| unknown3   | 4    | int32u | Large integer that increases in steps of folder header size (=44) Maybe the folder header position in HEYEX database not in this file? Possibly related to the folder header unknown4 value |
