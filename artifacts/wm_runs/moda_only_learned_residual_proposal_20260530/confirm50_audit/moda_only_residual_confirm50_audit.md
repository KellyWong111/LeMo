# MoDA-Only Residual Confirm50 Audit

No bsl, no selector-v3, no risk-controlled integration. This is a paired audit of learned residual proposal only.

## Aggregate Metrics

|index_set|method|scale|top1|top3|top5|oracle|near-miss|fixed|harmed|net|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|confirm50|raw_moda|0.0|50.00|53.33|56.67|56.67|4||||
|confirm50|residual_calibrated_cost|0.1|50.00|53.33|53.33|56.67|4|0|0|0|
|confirm50|residual_calibrated_cost|0.2|50.00|53.33|55.00|56.67|4|0|0|0|
|confirm50|residual_calibrated_cost|0.25|50.00|53.33|55.00|56.67|4|0|0|0|
|confirm50|residual_calibrated_cost|0.3|50.00|53.33|55.00|56.67|4|0|0|0|
|confirm50|residual_calibrated_cost|0.4|51.67|53.33|55.00|56.67|3|1|0|1|
|confirm50|residual_calibrated_cost|0.5|51.67|53.33|55.00|56.67|3|1|0|1|
|confirm50|residual_raw_cost|0.1|50.00|53.33|53.33|56.67|4|0|0|0|
|confirm50|residual_raw_cost|0.2|50.00|53.33|55.00|56.67|4|0|0|0|
|confirm50|residual_raw_cost|0.25|50.00|53.33|56.67|56.67|4|0|0|0|
|confirm50|residual_raw_cost|0.3|50.00|53.33|56.67|56.67|4|0|0|0|
|confirm50|residual_raw_cost|0.4|51.67|53.33|56.67|56.67|3|1|0|1|
|confirm50|residual_raw_cost|0.5|51.67|53.33|56.67|56.67|3|1|0|1|
|confirm50|raw_moda|0.0|46.00|56.00|60.00|60.00|14||||
|confirm50|residual_calibrated_cost|0.1|48.00|55.00|57.00|60.00|12|2|0|2|
|confirm50|residual_calibrated_cost|0.2|48.00|56.00|58.00|60.00|12|2|0|2|
|confirm50|residual_calibrated_cost|0.25|49.00|56.00|59.00|60.00|11|3|0|3|
|confirm50|residual_calibrated_cost|0.3|49.00|56.00|59.00|60.00|11|3|0|3|
|confirm50|residual_calibrated_cost|0.4|50.00|56.00|59.00|60.00|10|4|0|4|
|confirm50|residual_calibrated_cost|0.5|50.00|56.00|59.00|60.00|10|4|0|4|
|confirm50|residual_raw_cost|0.1|46.00|56.00|57.00|60.00|14|0|0|0|
|confirm50|residual_raw_cost|0.2|46.00|56.00|58.00|60.00|14|0|0|0|
|confirm50|residual_raw_cost|0.25|46.00|56.00|60.00|60.00|14|0|0|0|
|confirm50|residual_raw_cost|0.3|46.00|56.00|60.00|60.00|14|0|0|0|
|confirm50|residual_raw_cost|0.4|47.00|56.00|60.00|60.00|13|1|0|1|
|confirm50|residual_raw_cost|0.5|47.00|56.00|60.00|60.00|13|1|0|1|
|confirm50|raw_moda|0.0|40.00|60.00|65.00|65.00|10||||
|confirm50|residual_calibrated_cost|0.1|45.00|57.50|62.50|65.00|8|2|0|2|
|confirm50|residual_calibrated_cost|0.2|45.00|60.00|62.50|65.00|8|2|0|2|
|confirm50|residual_calibrated_cost|0.25|47.50|60.00|65.00|65.00|7|3|0|3|
|confirm50|residual_calibrated_cost|0.3|47.50|60.00|65.00|65.00|7|3|0|3|
|confirm50|residual_calibrated_cost|0.4|47.50|60.00|65.00|65.00|7|3|0|3|
|confirm50|residual_calibrated_cost|0.5|47.50|60.00|65.00|65.00|7|3|0|3|
|confirm50|residual_raw_cost|0.1|40.00|60.00|62.50|65.00|10|0|0|0|
|confirm50|residual_raw_cost|0.2|40.00|60.00|62.50|65.00|10|0|0|0|
|confirm50|residual_raw_cost|0.25|40.00|60.00|65.00|65.00|10|0|0|0|
|confirm50|residual_raw_cost|0.3|40.00|60.00|65.00|65.00|10|0|0|0|
|confirm50|residual_raw_cost|0.4|40.00|60.00|65.00|65.00|10|0|0|0|
|confirm50|residual_raw_cost|0.5|40.00|60.00|65.00|65.00|10|0|0|0|
|medium20|raw_moda|0.0|60.00|65.00|70.00|75.00|6||||
|medium20|residual_calibrated_cost|0.25|62.50|65.00|70.00|75.00|5|1|0|1|
|medium20|residual_raw_cost|0.25|62.50|65.00|65.00|75.00|5|1|0|1|

## Files

- `residual_eval_indices.csv`
- `moda_only_residual_confirm50_audit.csv`
- `residual_scale_sensitivity.csv`
- `residual_case_studies.csv`
