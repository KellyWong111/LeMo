# MoDA-Only Learned Residual Proposal

|method|scale|top1|oracle|success density|near-miss|
|---|---:|---:|---:|---:|---:|
|raw_moda|0.0|60.00|75.00|56.75|6|
|residual_calibrated_cost|0.25|65.00|72.50|57.50|3|
|residual_calibrated_cost|0.5|60.00|75.00|58.25|6|
|residual_calibrated_cost|1.0|60.00|75.00|56.75|6|
|residual_calibrated_cost|1.5|60.00|75.00|57.00|6|
|residual_raw_cost|0.25|62.50|75.00|57.25|5|
|residual_raw_cost|0.5|60.00|75.00|57.00|6|
|residual_raw_cost|1.0|60.00|72.50|56.75|5|
|residual_raw_cost|1.5|60.00|75.00|57.00|6|

## Verdict

Best top1 is 65.00 with method=residual_calibrated_cost scale=0.25.
