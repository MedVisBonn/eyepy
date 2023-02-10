# Image structure (20 + width * height * type_factor bytes)

| name     | size                         | type             | description                                                               |
| -------- | ---------------------------- | ---------------- | ------------------------------------------------------------------------- |
| size     | 4                            | int32u           | Size of the data                                                          |
| type     | 4                            | int32u           | Type of the data                                                          |
| n_values | 4                            | int32u           | Number of values                                                          |
| height   | 4                            | int32u           | Height of the image                                                       |
| width    | 4                            | int32u           | Width of the image                                                        |
| data     | width * height * type_factor | int8u / float16u | Image data - Localizer are stored as int8u and Bscans as a custom float16 |


## Known types

| type ID  | Description   | datatype |
| -------- | ------------- | -------- |
| 33620481 | Localizer NIR | int8u    |
| 35652097 | Bscan         | float16u |

The custom `float16u` used to store the Bscan data, has no sign, a 6-bit exponent und 10-bit mantissa.
