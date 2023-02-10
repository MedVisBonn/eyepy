# Textdata Structure (8 + n_strings * string_size * 2 bytes)

This structure for text data is used by different folder types to store data.

| name        | size                        | type   | description |
| ----------- | --------------------------- | ------ | ----------- |
| n_strings   | 4                           | int32u |             |
| string_size | 4                           | int32u |             |
| text        | n_strings * string_size * 2 | utf16  |             |

This structure for text data is used by different folder types to store data.
