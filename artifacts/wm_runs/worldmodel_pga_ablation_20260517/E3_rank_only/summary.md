# World-Model Planning Geometry Alignment

|variant|alpha|original|corrected|oracle|net|fixed|harmed|
|---|---:|---:|---:|---:|---:|---:|---:|
|vf05_mix20|0.25|56.7|63.3|83.3|6.7|8|0|
|vf05_mix20|0.50|56.7|67.5|83.3|10.8|16|3|
|vf05_mix20|0.75|56.7|67.5|83.3|10.8|19|6|
|vf05_mix20|1.00|56.7|60.8|83.3|4.2|16|11|
|vf05|0.25|56.7|57.5|81.7|0.8|3|2|
|vf05|0.50|56.7|63.3|81.7|6.7|11|3|
|vf05|0.75|56.7|60.0|81.7|3.3|10|6|
|vf05|1.00|56.7|52.5|81.7|-4.2|7|12|
|vf03_mix20|0.25|67.5|70.0|75.8|2.5|3|0|
|vf03_mix20|0.50|67.5|68.3|75.8|0.8|5|4|
|vf03_mix20|0.75|67.5|68.3|75.8|0.8|6|5|
|vf03_mix20|1.00|67.5|61.7|75.8|-5.8|4|11|

## Best
- vf03_mix20: alpha=0.25, original=67.5, corrected=70.0, oracle=75.8, fixed=3, harmed=0
- vf05: alpha=0.50, original=56.7, corrected=63.3, oracle=81.7, fixed=11, harmed=3
- vf05_mix20: alpha=0.50, original=56.7, corrected=67.5, oracle=83.3, fixed=16, harmed=3
