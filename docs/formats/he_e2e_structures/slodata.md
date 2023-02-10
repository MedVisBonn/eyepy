# Slo Data (100 bytes)

| name      | size | type    | description                                     |
| --------- | ---- | ------- | ----------------------------------------------- |
| unknown0  | 24   | byte    |                                                 |
| windate   | 8    | int64u  | ASCII(76) = L = OS ; ASCI(82) = R = OD          |
| transform | 24   | float32 |                                                 |
| rest      | 44   | bytes   | Is this constant or does the slodata size vary? |
