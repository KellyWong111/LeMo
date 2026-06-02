# World-Model Planning Geometry Alignment

|variant|alpha|original|corrected|oracle|net|fixed|harmed|
|---|---:|---:|---:|---:|---:|---:|---:|
|vf05_mix20|0.25|56.7|64.2|83.3|7.5|9|0|
|vf05_mix20|0.50|56.7|67.5|83.3|10.8|15|2|
|vf05_mix20|0.75|56.7|65.8|83.3|9.2|18|7|
|vf05_mix20|1.00|56.7|58.3|83.3|1.7|16|14|
|vf05|0.25|56.7|62.5|81.7|5.8|7|0|
|vf05|0.50|56.7|66.7|81.7|10.0|12|0|
|vf05|0.75|56.7|62.5|81.7|5.8|14|7|
|vf05|1.00|56.7|55.0|81.7|-1.7|11|13|
|vf03_mix20|0.25|67.5|69.2|75.8|1.7|2|0|
|vf03_mix20|0.50|67.5|66.7|75.8|-0.8|3|4|
|vf03_mix20|0.75|67.5|66.7|75.8|-0.8|7|8|
|vf03_mix20|1.00|67.5|60.0|75.8|-7.5|4|13|

## Best
- vf03_mix20: alpha=0.25, original=67.5, corrected=69.2, oracle=75.8, fixed=2, harmed=0
- vf05: alpha=0.50, original=56.7, corrected=66.7, oracle=81.7, fixed=12, harmed=0
- vf05_mix20: alpha=0.50, original=56.7, corrected=67.5, oracle=83.3, fixed=15, harmed=2
