# PAC-MoDA v2 Result Audit

## Verdict

Issue found. The old `legacy_rank_combined` result in `pac_moda_v2_full_n100_20260529` used all-candidate feature normalization inside `fit_combined_ranker`, so it touched validation/held feature distribution. I reran a corrected version with train-only normalization in `pac_moda_v2_full_n100_corrected_20260529`.

The corrected main result remains valid: `legacy_rank_combined` / rank-preserve PAC-MoDA v2 OOF `fixed=6`, `harmed=0`, `switches=8`, `stateroll-only recovered=6`.

Old `fixed=6, harmed=0, switches=9` should be deprecated and replaced by corrected `fixed=6, harmed=0, switches=8`.

## Audit Checks

- `episode_level_fixed_harmed_switches`: **pass**. eval_final_selection groups by (seed, episode); selected_rows length equals switches and fixed/harmed are counted once per switched episode.
- `loso_train_eval_separation`: **partial_pass_with_caveat**. Held seed is excluded from training data and threshold selection. Fixed gate is reused from splitA/splitB precomputed gate rows, not learned from held labels in the LOSO run. Original legacy normalization used all feature rows and is deprecated; corrected run uses train-only normalization.
- `calibrated_topk_nearmiss_order`: **pass**. rank_metrics orders by np.argsort(-scores), and near_miss checks labels[order[0]], so calibrated top-k/near-miss use calibrated order.
- `legacy_vs_full_definition`: **pass_with_required_caveat**. legacy_rank_combined is ranking/listwise/preserve without BCE; full_bce_pairwise_listwise_preserve is BCE plus pairwise/listwise/preserve. They must not be merged in text.
- `budget_feature_normalization`: **issue_found**. Budget generalization uses the same feature definitions and train-only normalization in its legacy fitter. n50 has no matching fixed gate grid, so deployment comparison is unavailable. Earlier old full/cost_calibration_v2 ranking variants used all-X normalization and are deprecated for strict deployment claims.

## Deprecated Outputs

- `pac_moda_v2_full_n100_20260529 legacy_rank_combined`: deprecated due to all-X normalization leakage in fit_combined_ranker
- `cost_calibration_v2_n100_20260529 pairwise/listwise/combined`: deprecated for strict split claims due to all-X normalization in ranking fitters; BCE remains train-normalized

## Corrected OOF Deployment

|method|fixed|harmed|switches|stateroll-only recovered|
|---|---:|---:|---:|---:|
|legacy_rank_combined corrected|6|0|8|6|
|full_bce_pairwise_listwise_preserve corrected|3|0|3|3|

## Corrected LOSO

Corrected LOSO legacy total: fixed=3, harmed=0, switches=8, stateroll-only recovered=3.

This means A/B OOF remains strong, but LOSO is weaker after strict train-only normalization. Report LOSO as robustness evidence, not as another fixed=6 claim.
