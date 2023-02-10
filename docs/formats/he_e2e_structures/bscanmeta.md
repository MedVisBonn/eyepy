# Bscan-Meta Structure (428 bytes)

The current Bscan-Meta structure builds on the implementation found in [LibE2E](https://github.com/neurodial/LibE2E/blob/d26d2d9db64c5f765c0241ecc22177bb0c440c87/E2E/dataelements/bscanmetadataelement.cpp#L75).

| name             | size | type    | description                                                                 |
| ---------------- | ---- | ------- | --------------------------------------------------------------------------- |
| unknown          | 4    | int32u  |                                                                             |
| size_y           | 4    | int32u  | Bscan Height                                                                |
| size_x           | 4    | int32u  | Bscan Width                                                                 |
| start_x          | 4    | float32 | Start X coordinate                                                          |
| start_y          | 4    | float32 | Start Y coordinate                                                          |
| end_x            | 4    | float32 | End X coordinate                                                            |
| end_y            | 4    | float32 | End Y coordinate                                                            |
| zero1            | 4    | int32u  |                                                                             |
| unknown1         | 4    | float32 |                                                                             |
| scale_y          | 4    | float32 | Scale Y                                                                     |
| unknown2         | 4    | float32 |                                                                             |
| zero2            | 4    | int32u  |                                                                             |
| unknown3         | 8    | float32 |                                                                             |
| zero3            | 4    | int32u  |                                                                             |
| imgSizeWidth     | 4    | int32u  | This might differ from size_x, the actual meaning of this value is unclear. |
| n_bscans         | 4    | int32u  | Number of Bscans                                                            |
| current_bscan    | 4    | int32u  | Index of current Bscan                                                      |
| scan_pattern     | 4    | int32u  | Scan Pattern                                                                |
| center_x         | 4    | float32 | Center X coordinate                                                         |
| center_y         | 4    | float32 | Center Y coordinate                                                         |
| unknown4         | 4    | int32u  |                                                                             |
| acquisition_time | 8    | int64u  | Acquisition Time                                                            |
| num_ave          | 4    | int32u  | Number of Averages                                                          |
| quality          | 4    | float32 | Quality                                                                     |
| rest             | 324  | bytes   |                                                                             |
