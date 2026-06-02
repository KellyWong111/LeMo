# PAC-MoDA MoDA-Only Calibrated Planner

This report evaluates PAC-MoDA as a direct MoDA/stateroll-only planner. The main table does not use bsl success/failure, fixed, harmed, or net. Each method selects the top candidate only from the stateroll/MoDA candidate pool.

## OOF Summary

|method|top1|top3|top5|top10|top30|AUC|first-success rank|near-miss|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|raw_stateroll_cost|55.33|63.00|66.17|71.50|78.33|0.494|3.38|138|
|bce|54.17|63.17|68.17|73.33|78.33|0.688|3.00|145|
|bce_pairwise|56.00|62.50|66.50|71.17|78.33|0.682|3.27|134|
|bce_listwise|54.00|65.67|69.17|73.83|78.33|0.691|2.74|146|
|bce_preserve|50.00|66.33|71.83|75.50|78.33|0.680|2.47|170|
|bce_pairwise_preserve|56.00|62.83|67.50|72.50|78.33|0.683|3.08|134|
|bce_listwise_preserve|51.50|66.50|71.00|75.17|78.33|0.685|2.53|161|
|bce_pairwise_listwise|56.00|62.67|67.83|72.00|78.33|0.692|3.10|134|
|full_bce_pairwise_listwise_preserve|56.33|63.50|68.83|72.33|78.33|0.690|2.97|132|
|legacy_rank_combined|56.33|62.83|68.00|71.67|78.33|0.572|3.11|132|

## Per-Split Direct MoDA-Only Planning

|split|method|top1|top3|top5|top10|AUC|first-success rank|near-miss|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|raw_stateroll_cost|55.0|63.3|67.3|72.0|0.498|3.16|70|
|splitA_train42_44_val45_47|bce|54.0|64.3|68.3|74.0|0.686|2.86|73|
|splitA_train42_44_val45_47|bce_pairwise|55.3|62.7|67.7|71.0|0.681|3.11|69|
|splitA_train42_44_val45_47|bce_listwise|55.3|66.3|69.7|74.3|0.690|2.63|69|
|splitA_train42_44_val45_47|bce_preserve|52.3|68.7|72.3|75.3|0.675|2.40|78|
|splitA_train42_44_val45_47|bce_pairwise_preserve|55.3|63.7|69.0|74.0|0.677|2.78|69|
|splitA_train42_44_val45_47|bce_listwise_preserve|52.7|67.7|71.3|74.7|0.681|2.51|77|
|splitA_train42_44_val45_47|bce_pairwise_listwise|55.3|63.3|69.0|73.7|0.689|2.85|69|
|splitA_train42_44_val45_47|full_bce_pairwise_listwise_preserve|56.0|64.3|70.3|73.7|0.684|2.71|67|
|splitA_train42_44_val45_47|legacy_rank_combined|55.7|63.7|70.0|72.3|0.531|2.86|68|
|splitB_train45_47_val42_44|raw_stateroll_cost|55.7|62.7|65.0|71.0|0.490|3.59|68|
|splitB_train45_47_val42_44|bce|54.3|62.0|68.0|72.7|0.689|3.13|72|
|splitB_train45_47_val42_44|bce_pairwise|56.7|62.3|65.3|71.3|0.684|3.43|65|
|splitB_train45_47_val42_44|bce_listwise|52.7|65.0|68.7|73.3|0.693|2.86|77|
|splitB_train45_47_val42_44|bce_preserve|47.7|64.0|71.3|75.7|0.684|2.54|92|
|splitB_train45_47_val42_44|bce_pairwise_preserve|56.7|62.0|66.0|71.0|0.689|3.39|65|
|splitB_train45_47_val42_44|bce_listwise_preserve|50.3|65.3|70.7|75.7|0.689|2.55|84|
|splitB_train45_47_val42_44|bce_pairwise_listwise|56.7|62.0|66.7|70.3|0.695|3.35|65|
|splitB_train45_47_val42_44|full_bce_pairwise_listwise_preserve|56.7|62.7|67.3|71.0|0.695|3.24|65|
|splitB_train45_47_val42_44|legacy_rank_combined|57.0|62.0|66.0|71.0|0.613|3.36|64|

## Interpretation

Raw stateroll/MoDA direct top1 is 55.33. The best MoDA-only direct top1 in this evaluation is 56.33 from `full_bce_pairwise_listwise_preserve`. The best global candidate AUC is 0.692 from `bce_pairwise_listwise`, but its direct top1 is 56.00.

This means current calibration improves MoDA candidate-level ranking evidence, especially AUC and sometimes top-k/near-miss, but it does not yet turn MoDA/stateroll into an independently strong top1 planner. The final 82.0/83.17 results should therefore be described as bsl-relative integration results, not as proof of a fully standalone MoDA-native planner.

`selector-v3 balanced` is not reported as a global MoDA-only ranking improvement because its localized raw-cost activation changes the operating regime, not the global stateroll candidate ordering.
