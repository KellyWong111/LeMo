# PAC-MoDA Cost Calibration v2 n100

Frozen candidate pools. No encoder/predictor/candidate-pool changes. This compares BCE, pairwise, listwise, and combined planning-aware linear calibration heads.
Deployment is restricted to the previously fixed precision gates. Thresholds are selected on train seeds only; validation labels are not used for threshold or gate choice.

## Main Table: Raw Stateroll Cost vs Calibrated Stateroll Score

|split|method|score|candidate AUC|first-success rank mean|first-success rank median|top1|top3|top5|top10|top30|near-miss|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bce|raw stateroll cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|
|splitA_train42_44_val45_47|bce|calibrated stateroll score|0.687|2.88|1.00|54.7|64.3|68.0|74.0|78.3|71|
|splitA_train42_44_val45_47|pairwise|raw stateroll cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|
|splitA_train42_44_val45_47|pairwise|calibrated stateroll score|0.572|3.11|1.00|55.3|63.0|68.0|71.3|78.3|69|
|splitA_train42_44_val45_47|listwise|raw stateroll cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|
|splitA_train42_44_val45_47|listwise|calibrated stateroll score|0.622|2.74|1.00|52.3|67.7|69.7|74.0|78.3|78|
|splitA_train42_44_val45_47|combined|raw stateroll cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|
|splitA_train42_44_val45_47|combined|calibrated stateroll score|0.537|2.86|1.00|55.3|63.7|70.0|72.7|78.3|69|
|splitB_train45_47_val42_44|bce|raw stateroll cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|
|splitB_train45_47_val42_44|bce|calibrated stateroll score|0.689|3.16|1.00|54.0|62.3|67.0|72.0|78.3|73|
|splitB_train45_47_val42_44|pairwise|raw stateroll cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|
|splitB_train45_47_val42_44|pairwise|calibrated stateroll score|0.609|3.50|1.00|56.3|62.0|65.3|71.0|78.3|66|
|splitB_train45_47_val42_44|listwise|raw stateroll cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|
|splitB_train45_47_val42_44|listwise|calibrated stateroll score|0.659|2.57|1.00|51.7|65.7|71.7|75.3|78.3|80|
|splitB_train45_47_val42_44|combined|raw stateroll cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|
|splitB_train45_47_val42_44|combined|calibrated stateroll score|0.609|3.36|1.00|57.3|61.7|66.3|71.0|78.3|63|

## Candidate-Level Validation Metrics: All Pools

|split|method|pool|raw AUC|cal AUC|raw first rank|cal first rank|raw top1|cal top1|raw top10|cal top10|raw near-miss|cal near-miss|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bce|bsl|0.662|0.914|1.64|1.56|82.0|81.0|86.7|87.3|20|23|
|splitA_train42_44_val45_47|bce|stateroll|0.498|0.687|3.16|2.88|55.0|54.7|72.0|74.0|70|71|
|splitA_train42_44_val45_47|bce|union|0.649|0.853|3.86|3.58|75.7|81.7|86.0|88.3|58|40|
|splitA_train42_44_val45_47|pairwise|bsl|0.662|0.735|1.64|1.63|82.0|82.3|86.7|86.3|20|19|
|splitA_train42_44_val45_47|pairwise|stateroll|0.498|0.572|3.16|3.11|55.0|55.3|72.0|71.3|70|69|
|splitA_train42_44_val45_47|pairwise|union|0.649|0.736|3.86|3.44|75.7|76.3|86.0|87.3|58|56|
|splitA_train42_44_val45_47|listwise|bsl|0.662|0.715|1.64|1.39|82.0|79.7|86.7|87.7|20|27|
|splitA_train42_44_val45_47|listwise|stateroll|0.498|0.622|3.16|2.74|55.0|52.3|72.0|74.0|70|78|
|splitA_train42_44_val45_47|listwise|union|0.649|0.686|3.86|3.52|75.7|72.7|86.0|87.7|58|67|
|splitA_train42_44_val45_47|combined|bsl|0.662|0.372|1.64|1.45|82.0|81.0|86.7|87.7|20|23|
|splitA_train42_44_val45_47|combined|stateroll|0.498|0.537|3.16|2.86|55.0|55.3|72.0|72.7|70|69|
|splitA_train42_44_val45_47|combined|union|0.649|0.670|3.86|3.53|75.7|82.0|86.0|88.7|58|39|
|splitB_train45_47_val42_44|bce|bsl|0.590|0.933|1.73|1.65|80.0|79.0|85.7|86.3|25|28|
|splitB_train45_47_val42_44|bce|stateroll|0.490|0.689|3.59|3.16|55.7|54.0|71.0|72.0|68|73|
|splitB_train45_47_val42_44|bce|union|0.613|0.857|4.72|3.44|72.7|78.7|82.7|87.0|65|47|
|splitB_train45_47_val42_44|pairwise|bsl|0.590|0.654|1.73|1.72|80.0|79.7|85.7|85.7|25|26|
|splitB_train45_47_val42_44|pairwise|stateroll|0.490|0.609|3.59|3.50|55.7|56.3|71.0|71.0|68|66|
|splitB_train45_47_val42_44|pairwise|union|0.613|0.692|4.72|2.64|72.7|75.0|82.7|89.0|65|58|
|splitB_train45_47_val42_44|listwise|bsl|0.590|0.701|1.73|1.45|80.0|78.7|85.7|87.0|25|29|
|splitB_train45_47_val42_44|listwise|stateroll|0.490|0.659|3.59|2.57|55.7|51.7|71.0|75.3|68|80|
|splitB_train45_47_val42_44|listwise|union|0.613|0.729|4.72|3.63|72.7|74.3|82.7|86.0|65|60|
|splitB_train45_47_val42_44|combined|bsl|0.590|0.301|1.73|1.70|80.0|79.7|85.7|86.0|25|26|
|splitB_train45_47_val42_44|combined|stateroll|0.490|0.609|3.59|3.36|55.7|57.3|71.0|71.0|68|63|
|splitB_train45_47_val42_44|combined|union|0.613|0.661|4.72|3.52|72.7|80.0|82.7|86.7|65|43|

## Fixed Precision Gate Final Selection

|split|method|gate|bsl top1|calibrated selector top1|gate selected|gate st-only|gate bsl FP|threshold|fixed|harmed|switches|st-only recovered|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bce|extratrees top10+st_gap_bottom20+AND|82.0|82.7|9|6|3|0.8600|2|0|3|2|
|splitA_train42_44_val45_47|pairwise|extratrees top10+st_gap_bottom20+AND|82.0|82.0|9|6|3|1.7710|0|0|0|0|
|splitA_train42_44_val45_47|listwise|extratrees top10+st_gap_bottom20+AND|82.0|82.0|9|6|3|0.7049|0|0|0|0|
|splitA_train42_44_val45_47|combined|extratrees top10+st_gap_bottom20+AND|82.0|83.3|9|6|3|0.0095|4|0|5|4|
|splitB_train45_47_val42_44|bce|randomforest top10+abs_gap_top10+AND|80.0|80.0|8|3|4|0.8459|0|0|0|0|
|splitB_train45_47_val42_44|pairwise|randomforest top10+abs_gap_top10+AND|80.0|80.0|8|3|4|1.5779|0|0|0|0|
|splitB_train45_47_val42_44|listwise|randomforest top10+abs_gap_top10+AND|80.0|80.0|8|3|4|0.4195|0|0|0|0|
|splitB_train45_47_val42_44|combined|randomforest top10+abs_gap_top10+AND|80.0|80.7|8|3|4|-0.1217|2|0|4|2|

## OOF Deployment Totals

|method|fixed|harmed|switches|stateroll-only recovered|
|---|---:|---:|---:|---:|
|bce|2|0|3|2|
|pairwise|0|0|0|0|
|listwise|0|0|0|0|
|combined|6|0|9|6|
