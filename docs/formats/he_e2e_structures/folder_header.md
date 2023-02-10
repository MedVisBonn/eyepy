# Folder Header Structure (44 bytes)

| Name       | Size | Type   | Description                                                                                                                                  |
| ---------- | ---- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| pos        | 4    | int32u | Position of the folder (All 512 folder header in a chunk are stored sequentially, refering to the data that follows after this header block) |
| start      | 4    | int32u | Start of the data container, after the header block                                                                                          |
| size       | 4    | int32u | Size of the data container                                                                                                                   |
| unknown0   | 4    | int32u | Always 0 (b'\x00\x00')? At leat in our data                                                                                                  |
| patient_id | 4    | int32s | Patient ID                                                                                                                                   |
| study_id   | 4    | int32s | Study ID                                                                                                                                     |
| series_id  | 4    | int32s | Series ID                                                                                                                                    |
| slice_id   | 4    | int32s | Slice ID                                                                                                                                     |
| ind        | 2    | int16u | 0 for enface and 1 for bscan for image containers                                                                                            |
| unknown1   | 2    | int16u |                                                                                                                                       |
| type       | 4    | int32u | Type of the data container                                                                                                                   |
| unknown2   | 4    | int32u | Large integer possibly related to data_container.unknown5. Maybe the position in HEYEX DB?                                                   |
