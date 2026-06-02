# World-Model Planning Geometry Alignment

|variant|alpha|original|corrected|oracle|net|fixed|harmed|
|---|---:|---:|---:|---:|---:|---:|---:|
|vf05_mix20|0.25|56.7|63.3|83.3|6.7|9|1|
|vf05_mix20|0.50|56.7|66.7|83.3|10.0|15|3|
|vf05_mix20|0.75|56.7|67.5|83.3|10.8|18|5|
|vf05_mix20|1.00|56.7|56.7|83.3|0.0|14|14|
|vf05|0.25|56.7|60.8|81.7|4.2|5|0|
|vf05|0.50|56.7|64.2|81.7|7.5|10|1|
|vf05|0.75|56.7|62.5|81.7|5.8|14|7|
|vf05|1.00|56.7|51.7|81.7|-5.0|8|14|
|vf03_mix20|0.25|67.5|70.0|75.8|2.5|3|0|
|vf03_mix20|0.50|67.5|66.7|75.8|-0.8|3|4|
|vf03_mix20|0.75|67.5|66.7|75.8|-0.8|5|6|
|vf03_mix20|1.00|67.5|62.5|75.8|-5.0|4|10|

## Best
- vf03_mix20: alpha=0.25, original=67.5, corrected=70.0, oracle=75.8, fixed=3, harmed=0
- vf05: alpha=0.50, original=56.7, corrected=64.2, oracle=81.7, fixed=10, harmed=1
- vf05_mix20: alpha=0.50, original=56.7, corrected=66.7, oracle=83.3, fixed=15, harmed=3
