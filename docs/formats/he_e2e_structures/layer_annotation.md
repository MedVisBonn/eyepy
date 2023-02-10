# Layer Annotation Structure (16 + 4 * width bytes)

| name     | size      | type    | description                         |
| -------- | --------- | ------- | ----------------------------------- |
| unknown0 | 4         | int32u  |                                     |
| id       | 4         | int32u  | Layer ID                            |
| unknown1 | 4         | int32u  |                                     |
| width    | 4         | int32u  | Width of the layer                  |
| data     | 4 * width | float32 | Segmentation data (width * 4 bytes) |
