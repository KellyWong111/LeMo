# MoDA Cost Calibration Head n100

Frozen candidate pools. No encoder/predictor/candidate-pool changes. A small linear calibration head is trained on candidate features only.

## Main Table: Raw Stateroll Cost vs Calibrated Stateroll Score

|split|score|candidate AUC|first-success rank mean|first-success rank median|top1|top3|top5|top10|top30|near-miss|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|raw stateroll cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|
|splitA_train42_44_val45_47|calibrated stateroll score|0.687|2.88|1.00|54.7|64.3|68.0|74.0|78.3|70|
|splitB_train45_47_val42_44|raw stateroll cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|
|splitB_train45_47_val42_44|calibrated stateroll score|0.689|3.16|1.00|54.0|62.3|67.0|72.0|78.3|68|

## Candidate-Level Validation Metrics: All Pools

|split|pool|raw AUC|cal AUC|raw first rank|cal first rank|raw top1|cal top1|raw top10|cal top10|raw near-miss|cal near-miss|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bsl|0.662|0.914|1.64|1.56|82.0|81.0|86.7|87.3|20|20|
|splitA_train42_44_val45_47|stateroll|0.498|0.687|3.16|2.88|55.0|54.7|72.0|74.0|70|70|
|splitA_train42_44_val45_47|union|0.649|0.853|3.86|3.58|75.7|81.7|86.0|88.3|39|39|
|splitB_train45_47_val42_44|bsl|0.590|0.933|1.73|1.65|80.0|79.0|85.7|86.3|25|25|
|splitB_train45_47_val42_44|stateroll|0.490|0.689|3.59|3.16|55.7|54.0|71.0|72.0|68|68|
|splitB_train45_47_val42_44|union|0.613|0.857|4.72|3.44|72.7|78.7|82.7|87.0|43|43|

## Fixed Precision Gate Final Selection

|split|gate|bsl top1|calibrated selector top1|gate selected|gate st-only|gate bsl FP|threshold|fixed|harmed|switches|st-only recovered|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|extratrees top10+st_gap_bottom20+AND|82.0|82.7|9|6|3|0.8600|2|0|3|2|
|splitB_train45_47_val42_44|randomforest top10+abs_gap_top10+AND|80.0|80.0|8|3|4|0.8459|0|0|0|0|

OOF fixed=2, harmed=0, switches=3, stateroll-only recovered=2.
