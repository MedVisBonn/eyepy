# Measurement Structure (68 bytes)

| name            | size | type    | description           |
| --------------- | ---- | ------- | --------------------- |
| laterality      | 1    | ascii   | L for OS and R for OD |
| c_cureve_mm     | 8    | float64 |                       |
| refraction      | 8    | float64 |                       |
| cylinder_dpt    | 8    | float64 |                       |
| axis_deg        | 8    | float64 |                       |
| pupil_size      | 8    | float64 |                       |
| iop_mmHg        | 8    | float64 |                       |
| vfield_mean     | 8    | float64 |                       |
| vfield_var      | 8    | float64 |                       |
| corrective_lens | 2    | int16u  |                       |
| rest            | 1    | bytes   |                       |
