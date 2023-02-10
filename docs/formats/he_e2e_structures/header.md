# Header Structure (52 bytes)

| name        | size | type    | description                   |
| ----------- | ---- | ------- | ----------------------------- |
| magic2      | 12   | ascii   | Name of the version           |
| version     | 4    | float32 | Version of the file           |
| unknown0    | 20   |         |                               |
| num_entries | 4    | int32u  | Number of entries in the file |
| current     | 4    | int32u  | Current entry                 |
| prev        | 4    | int32u  | Previous entry                |
| unknown1    | 4    | int32u  |                               |
