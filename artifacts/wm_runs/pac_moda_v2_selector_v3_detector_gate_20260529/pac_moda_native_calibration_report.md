# PAC-MoDA Native Calibration Report

This report frames PAC-MoDA as MoDA-native planning-aware candidate utility calibration. The primary object is the MoDA/stateroll candidate pool. The strong baseline is used only as an evaluation reference, not as an algorithmic dependency.

Key wording: global raw MoDA cost fails, localized calibrated cost works.

## MoDA-Native Global Candidate Ranking

This table only includes scores that are meaningful for global MoDA candidate ranking. `localized_raw_cost_score` and `selector_v3_balanced_score` are not included here because they are raw-cost-equivalent orderings in the full stateroll pool; multiplying raw cost by a constant does not change AUC, first-success rank, top-k recall, or near-miss count.

|split|score|AUC|first rank mean|first rank median|top1|top3|top5|top10|top30|near-miss|episodes with success|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|splitA_train42_44_val45_47|raw_stateroll_cost|0.498|3.16|1.00|55.0|63.3|67.3|72.0|78.3|70|235|
|splitA_train42_44_val45_47|bce_calibrated_utility|0.690|2.88|1.00|54.7|64.3|68.3|73.7|78.3|71|235|
|splitA_train42_44_val45_47|rank_preserve_utility|0.531|2.86|1.00|55.7|63.7|70.0|72.3|78.3|68|235|
|splitB_train45_47_val42_44|raw_stateroll_cost|0.490|3.59|1.00|55.7|62.7|65.0|71.0|78.3|68|235|
|splitB_train45_47_val42_44|bce_calibrated_utility|0.690|3.26|1.00|55.3|62.0|67.3|71.3|78.3|69|235|
|splitB_train45_47_val42_44|rank_preserve_utility|0.613|3.36|1.00|57.0|62.0|66.0|71.0|78.3|64|235|

Interpretation: global raw MoDA cost is poorly aligned with task success (AUC near 0.5). BCE calibration improves global candidate-level AUC, while rank-preserve utility gives more localized improvements in top1 / near-miss behavior. These are MoDA-pool ranking metrics, independent of any bsl deployment rule.

## Localized Activation Regime

Localized activation is a deployment-level regime, not a global ranking metric. Detector-v2 does not make the raw MoDA cost globally better; it restricts decisions to episodes where the raw MoDA cost is reliable enough to be useful. Outside the selected regime, the strong baseline comparison keeps bsl as the evaluation reference.

|source|gate type|score used|mode|top1|fixed|harmed|net|switches|st-only recovered|
|---|---|---|---|---:|---:|---:|---:|---:|---:|
|strong_baseline_comparison|corrected conservative gate|rank_preserve_utility|conservative_harmed0|82.00|6|0|6|8|6|
|strong_baseline_comparison|detector-v2 gate|rank_preserve_utility|detector_v2_safe|82.00|6|0|6|8|6|
|strong_baseline_comparison|detector-v2 gate|localized raw MoDA cost|balanced_harmed2|83.17|15|2|13|59|9|
|selector_v3_ablation|global raw MoDA cost|raw_cost|global_raw_cost|80.00|0|0|0|0|0|
|selector_v3_ablation|fixed precision gate|raw_cost|fixed_precision_raw|81.67|6|2|4|17|6|
|selector_v3_ablation|detector-v2 gate|localized raw MoDA cost|detector_v2_raw_balanced|83.17|15|2|13|59|9|
|selector_v3_ablation|detector-v2 gate|BCE calibrated utility|detector_v2_bce|82.17|9|2|7|59|6|

Interpretation: global raw-cost override is not the method. The key result is that detector-v2 localized activation changes the operating regime: the same raw MoDA cost that is unreliable globally can recover useful candidates when applied only inside the opportunity-aware region. Conservative rank-preserve remains the harmed-free operating point; selector-v3 balanced is the higher-gain operating point with harmed=2.

## Strong Baseline Comparison

The bsl planner is a strong evaluation baseline and safety reference, not an algorithmic dependency for MoDA-native utility calibration.

See `pac_moda_strong_baseline_comparison.md/csv/json` for the separated baseline comparison.

## Clean Claim

PAC-MoDA has two complementary components: first, MoDA-native utility calibration improves global candidate ranking within the MoDA pool; second, opportunity-aware localized activation identifies regimes where raw MoDA cost can be safely reused for deployment. The baseline is only used as a strong evaluation reference.
