# World-Model Planning Geometry Alignment

|variant|alpha|original|corrected|oracle|net|fixed|harmed|
|---|---:|---:|---:|---:|---:|---:|---:|
|vf05_mix20|0.25|56.7|62.5|83.3|5.8|9|2|
|vf05_mix20|0.50|56.7|68.3|83.3|11.7|17|3|
|vf05_mix20|0.75|56.7|65.8|83.3|9.2|18|7|
|vf05_mix20|1.00|56.7|59.2|83.3|2.5|17|14|
|vf05|0.25|56.7|61.7|81.7|5.0|7|1|
|vf05|0.50|56.7|65.8|81.7|9.2|13|2|
|vf05|0.75|56.7|62.5|81.7|5.8|15|8|
|vf05|1.00|56.7|50.0|81.7|-6.7|9|17|
|vf03_mix20|0.25|67.5|70.0|75.8|2.5|3|0|
|vf03_mix20|0.50|67.5|67.5|75.8|0.0|5|5|
|vf03_mix20|0.75|67.5|65.8|75.8|-1.7|6|8|
|vf03_mix20|1.00|67.5|63.3|75.8|-4.2|5|10|

## Best
- vf03_mix20: alpha=0.25, original=67.5, corrected=70.0, oracle=75.8, fixed=3, harmed=0
- vf05: alpha=0.50, original=56.7, corrected=65.8, oracle=81.7, fixed=13, harmed=2
- vf05_mix20: alpha=0.50, original=56.7, corrected=68.3, oracle=83.3, fixed=17, harmed=3
