# PAC-MoDA v2 Selector v3 Audit

Selector v3 uses detector-v2 gate `logistic fp_le_3_max30`, not the original fixed precision gate. The inherited report sentence saying fixed precision gates is stale and should be read as detector-v2 gate restricted deployment.

Thresholds are selected on train seeds only in `train_thresholds`. OOF harmed budgets are train-side budgets, so the best OOF row can have `harmed=2` even for `harmed_budget=0`.

Best current operating point:

|w_bce|w_rank|w_raw|candidate_topk|fixed|harmed|net|switches|st-only recovered|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.0|0.0|0.5|1|15|2|13|59|9|

Interpretation: this is a balanced/aggressive operating point, not the harmed=0 conservative result.
