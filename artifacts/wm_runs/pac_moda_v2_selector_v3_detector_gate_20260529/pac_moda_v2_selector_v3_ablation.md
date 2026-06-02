# PAC-MoDA v2 Selector v3 Ablation

|gate|score|harmed budget|fixed|harmed|net|switches|st-only recovered|approx top1|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|global|rank_preserve|2|20|3|17|56|17|83.83|
|global|rank_preserve|0|19|2|17|48|16|83.83|
|detector_v2|rank_preserve|0|15|2|13|59|9|83.17|
|detector_v2|rank_preserve|2|15|2|13|59|9|83.17|
|detector_v2|raw_cost|0|15|2|13|59|9|83.17|
|detector_v2|raw_cost|2|15|2|13|59|9|83.17|
|detector_v2|selector_v3_raw|0|15|2|13|59|9|83.17|
|detector_v2|selector_v3_raw|2|15|2|13|59|9|83.17|
|global|agreement_bce_rank|0|13|2|11|34|10|82.83|
|global|agreement_bce_rank|2|13|3|10|40|10|82.67|
|detector_v2|bce|0|9|2|7|59|6|82.17|
|detector_v2|bce|2|9|2|7|59|6|82.17|
|detector_v2|agreement_bce_rank|0|6|0|6|30|3|82.00|
|detector_v2|agreement_bce_rank|2|6|0|6|30|3|82.00|
|fixed_precision|rank_preserve|0|6|2|4|17|6|81.67|
|fixed_precision|rank_preserve|2|6|2|4|17|6|81.67|
|fixed_precision|raw_cost|0|6|2|4|17|6|81.67|
|fixed_precision|raw_cost|2|6|2|4|17|6|81.67|
|fixed_precision|selector_v3_raw|0|6|2|4|17|6|81.67|
|fixed_precision|selector_v3_raw|2|6|2|4|17|6|81.67|
|fixed_precision|bce|0|3|2|1|17|3|81.17|
|fixed_precision|bce|2|3|2|1|17|3|81.17|
|fixed_precision|agreement_bce_rank|0|2|1|1|8|2|81.17|
|fixed_precision|agreement_bce_rank|2|2|1|1|8|2|81.17|
|global|bce|0|1|0|1|1|1|81.17|
|global|raw_cost|0|0|0|0|2|0|81.00|
|global|selector_v3_raw|0|0|0|0|2|0|81.00|
|global|bce|2|1|3|-2|19|1|80.67|
|global|raw_cost|2|0|2|-2|7|0|80.67|
|global|selector_v3_raw|2|0|2|-2|7|0|80.67|

Key comparison: global raw-cost switching is included to show that raw cost is only viable when restricted by detector-v2 opportunity gating.
