# MoDA-Only Learned Residual Proposal

|method|scale|top1|oracle|success density|near-miss|
|---|---:|---:|---:|---:|---:|
|raw_moda|0.0|46.00|60.00|50.00|14|
|residual_calibrated_cost|0.25|49.00|60.00|50.00|11|
|residual_raw_cost|0.25|46.00|60.00|49.70|14|

## Verdict

Best top1 is 49.00 with method=residual_calibrated_cost scale=0.25.
