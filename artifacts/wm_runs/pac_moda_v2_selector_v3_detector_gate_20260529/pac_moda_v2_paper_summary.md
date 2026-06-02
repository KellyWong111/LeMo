# PAC-MoDA v2 Paper Summary

|mode|fixed|harmed|net|switches|st-only recovered|approx top1|
|---|---:|---:|---:|---:|---:|---:|
|conservative_rank_preserve|6|0|6|8|6|82.00|
|detector_v2_gate_rank_preserve|6|0|6|8|6|82.00|
|selector_v3_balanced|15|2|13|59|9|83.17|

Use `conservative_rank_preserve` as the harmed=0 safe result.
Use `selector_v3_balanced` as the aggressive/balanced operating point; it has harmed=2 and should not be described as harmed-free.
