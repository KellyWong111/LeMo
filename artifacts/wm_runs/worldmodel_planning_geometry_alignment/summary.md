# World-Model Planning Geometry Alignment

|variant|alpha|original|corrected|oracle|net|fixed|harmed|
|---|---:|---:|---:|---:|---:|---:|---:|
|vf05_mix20|0.25|56.7|64.2|83.3|7.5|9|0|
|vf05_mix20|0.50|56.7|69.2|83.3|12.5|15|0|
|vf05_mix20|0.75|56.7|65.8|83.3|9.2|17|6|
|vf05_mix20|1.00|56.7|60.8|83.3|4.2|17|12|
|vf05|0.25|56.7|59.2|81.7|2.5|4|1|
|vf05|0.50|56.7|62.5|81.7|5.8|10|3|
|vf05|0.75|56.7|59.2|81.7|2.5|12|9|
|vf05|1.00|56.7|52.5|81.7|-4.2|11|16|
|vf03_mix20|0.25|67.5|70.0|75.8|2.5|3|0|
|vf03_mix20|0.50|67.5|68.3|75.8|0.8|5|4|
|vf03_mix20|0.75|67.5|65.8|75.8|-1.7|5|7|
|vf03_mix20|1.00|67.5|62.5|75.8|-5.0|5|11|

## Best
- vf03_mix20: alpha=0.25, original=67.5, corrected=70.0, oracle=75.8, fixed=3, harmed=0
- vf05: alpha=0.50, original=56.7, corrected=62.5, oracle=81.7, fixed=10, harmed=3
- vf05_mix20: alpha=0.50, original=56.7, corrected=69.2, oracle=83.3, fixed=15, harmed=0
