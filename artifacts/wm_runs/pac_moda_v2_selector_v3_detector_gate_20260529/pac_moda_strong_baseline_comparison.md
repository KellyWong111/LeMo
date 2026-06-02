# PAC-MoDA Strong Baseline Comparison

The baseline is a strong evaluation reference. PAC-MoDA is not described as first running the baseline and then patching failures.

|method|top1|fixed|harmed|net|switches|st-only recovered|
|---|---:|---:|---:|---:|---:|---:|
|bsl|81.00|-|-|-|-|-|
|conservative_rank_preserve|82.0|6|0|6|8|6|
|detector_v2_gate_rank_preserve|82.0|6|0|6|8|6|
|selector_v3_balanced|83.16666666666667|15|2|13|59|9|
