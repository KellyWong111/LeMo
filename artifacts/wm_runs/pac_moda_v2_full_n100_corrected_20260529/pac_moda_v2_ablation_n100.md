# PAC-MoDA v2 Ablation n100

No encoder/predictor/world-model retraining. Candidate pools are frozen. Thresholds are selected on train seeds only.

## Stateroll Candidate-Level Metrics

|split|method|raw AUC|cal AUC|raw first rank|cal first rank|raw top1|cal top1|raw top3|cal top3|raw top5|cal top5|raw top10|cal top10|near-miss raw|near-miss cal|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bce|0.498|0.686|3.16|2.86|55.0|54.0|63.3|64.3|67.3|68.3|72.0|74.0|70|73|
|splitA_train42_44_val45_47|bce_pairwise|0.498|0.681|3.16|3.11|55.0|55.3|63.3|62.7|67.3|67.7|72.0|71.0|70|69|
|splitA_train42_44_val45_47|bce_listwise|0.498|0.690|3.16|2.63|55.0|55.3|63.3|66.3|67.3|69.7|72.0|74.3|70|69|
|splitA_train42_44_val45_47|bce_preserve|0.498|0.675|3.16|2.40|55.0|52.3|63.3|68.7|67.3|72.3|72.0|75.3|70|78|
|splitA_train42_44_val45_47|bce_pairwise_preserve|0.498|0.677|3.16|2.78|55.0|55.3|63.3|63.7|67.3|69.0|72.0|74.0|70|69|
|splitA_train42_44_val45_47|bce_listwise_preserve|0.498|0.681|3.16|2.51|55.0|52.7|63.3|67.7|67.3|71.3|72.0|74.7|70|77|
|splitA_train42_44_val45_47|bce_pairwise_listwise|0.498|0.689|3.16|2.85|55.0|55.3|63.3|63.3|67.3|69.0|72.0|73.7|70|69|
|splitA_train42_44_val45_47|full_bce_pairwise_listwise_preserve|0.498|0.684|3.16|2.71|55.0|56.0|63.3|64.3|67.3|70.3|72.0|73.7|70|67|
|splitA_train42_44_val45_47|legacy_rank_combined|0.498|0.531|3.16|2.86|55.0|55.7|63.3|63.7|67.3|70.0|72.0|72.3|70|68|
|splitB_train45_47_val42_44|bce|0.490|0.689|3.59|3.13|55.7|54.3|62.7|62.0|65.0|68.0|71.0|72.7|68|72|
|splitB_train45_47_val42_44|bce_pairwise|0.490|0.684|3.59|3.43|55.7|56.7|62.7|62.3|65.0|65.3|71.0|71.3|68|65|
|splitB_train45_47_val42_44|bce_listwise|0.490|0.693|3.59|2.86|55.7|52.7|62.7|65.0|65.0|68.7|71.0|73.3|68|77|
|splitB_train45_47_val42_44|bce_preserve|0.490|0.684|3.59|2.54|55.7|47.7|62.7|64.0|65.0|71.3|71.0|75.7|68|92|
|splitB_train45_47_val42_44|bce_pairwise_preserve|0.490|0.689|3.59|3.39|55.7|56.7|62.7|62.0|65.0|66.0|71.0|71.0|68|65|
|splitB_train45_47_val42_44|bce_listwise_preserve|0.490|0.689|3.59|2.55|55.7|50.3|62.7|65.3|65.0|70.7|71.0|75.7|68|84|
|splitB_train45_47_val42_44|bce_pairwise_listwise|0.490|0.695|3.59|3.35|55.7|56.7|62.7|62.0|65.0|66.7|71.0|70.3|68|65|
|splitB_train45_47_val42_44|full_bce_pairwise_listwise_preserve|0.490|0.695|3.59|3.24|55.7|56.7|62.7|62.7|65.0|67.3|71.0|71.0|68|65|
|splitB_train45_47_val42_44|legacy_rank_combined|0.490|0.613|3.59|3.36|55.7|57.0|62.7|62.0|65.0|66.0|71.0|71.0|68|64|

## Fixed-Gate Deployment

|split|method|bsl top1|selector top1|fixed|harmed|switches|st-only recovered|threshold|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|bce|82.0|82.7|2|0|3|2|1.8267|
|splitA_train42_44_val45_47|bce_pairwise|82.0|82.7|2|0|2|2|2.5586|
|splitA_train42_44_val45_47|bce_listwise|82.0|83.0|3|0|3|3|1.5459|
|splitA_train42_44_val45_47|bce_preserve|82.0|82.7|2|0|4|2|0.5721|
|splitA_train42_44_val45_47|bce_pairwise_preserve|82.0|83.0|3|0|3|3|1.2918|
|splitA_train42_44_val45_47|bce_listwise_preserve|82.0|82.7|2|0|4|2|0.5640|
|splitA_train42_44_val45_47|bce_pairwise_listwise|82.0|82.7|2|0|3|2|2.0705|
|splitA_train42_44_val45_47|full_bce_pairwise_listwise_preserve|82.0|83.0|3|0|3|3|0.9801|
|splitA_train42_44_val45_47|legacy_rank_combined|82.0|83.3|4|0|5|4|-0.0345|
|splitB_train45_47_val42_44|bce|80.0|80.0|0|0|0|0|1.7299|
|splitB_train45_47_val42_44|bce_pairwise|80.0|80.0|0|0|0|0|2.4155|
|splitB_train45_47_val42_44|bce_listwise|80.0|80.0|0|0|1|0|1.4520|
|splitB_train45_47_val42_44|bce_preserve|80.0|80.0|0|0|1|0|0.4679|
|splitB_train45_47_val42_44|bce_pairwise_preserve|80.0|80.0|0|0|0|0|1.1645|
|splitB_train45_47_val42_44|bce_listwise_preserve|80.0|80.0|0|0|1|0|0.4489|
|splitB_train45_47_val42_44|bce_pairwise_listwise|80.0|80.0|0|0|0|0|1.9334|
|splitB_train45_47_val42_44|full_bce_pairwise_listwise_preserve|80.0|80.0|0|0|0|0|0.8622|
|splitB_train45_47_val42_44|legacy_rank_combined|80.0|80.7|2|0|3|2|-0.0552|

## OOF Totals

|method|fixed|harmed|switches|st-only recovered|
|---|---:|---:|---:|---:|
|bce|2|0|3|2|
|bce_listwise|3|0|4|3|
|bce_listwise_preserve|2|0|5|2|
|bce_pairwise|2|0|2|2|
|bce_pairwise_listwise|2|0|3|2|
|bce_pairwise_preserve|3|0|3|3|
|bce_preserve|2|0|5|2|
|full_bce_pairwise_listwise_preserve|3|0|3|3|
|legacy_rank_combined|6|0|8|6|
