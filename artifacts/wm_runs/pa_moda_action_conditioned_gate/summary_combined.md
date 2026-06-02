# PA-MoDA Action-Conditioned Rollout Gate

|mode|variant|beta|n|original|corrected|oracle|fixed|harmed|net_gain|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|combined|vf03_mix20|0.05|6|66.7|66.7|75.8|0|0|0.0|
|combined|vf03_mix20|0.10|6|66.7|66.7|75.8|0|0|0.0|
|combined|vf03_mix20|0.25|6|66.7|66.7|75.8|0|0|0.0|
|combined|vf03_mix20|0.50|6|66.7|66.7|75.8|0|0|0.0|
|combined|vf03_mix20|1.00|6|66.7|66.7|75.8|0|0|0.0|
|combined|vf05|0.05|6|56.7|56.7|81.7|0|0|0.0|
|combined|vf05|0.10|6|56.7|56.7|81.7|0|0|0.0|
|combined|vf05|0.25|6|56.7|57.5|81.7|1|0|0.8|
|combined|vf05|0.50|6|56.7|57.5|81.7|1|0|0.8|
|combined|vf05|1.00|6|56.7|56.7|81.7|1|1|0.0|
|combined|vf05_mix20|0.05|6|56.7|57.5|83.3|1|0|0.8|
|combined|vf05_mix20|0.10|6|56.7|57.5|83.3|1|0|0.8|
|combined|vf05_mix20|0.25|6|56.7|57.5|83.3|1|0|0.8|
|combined|vf05_mix20|0.50|6|56.7|57.5|83.3|1|0|0.8|
|combined|vf05_mix20|1.00|6|56.7|56.7|83.3|1|1|0.0|

## Best by variant

- combined vf03_mix20: beta=0.05, original=66.7, corrected=66.7, oracle=75.8, fixed=0, harmed=0, net_gain=0.0
- combined vf05: beta=0.25, original=56.7, corrected=57.5, oracle=81.7, fixed=1, harmed=0, net_gain=0.8
- combined vf05_mix20: beta=0.05, original=56.7, corrected=57.5, oracle=83.3, fixed=1, harmed=0, net_gain=0.8
