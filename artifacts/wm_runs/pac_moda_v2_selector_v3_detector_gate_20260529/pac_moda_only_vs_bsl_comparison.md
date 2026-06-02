# PAC-MoDA MoDA-Only vs Strong Baseline Comparison

This table separates direct MoDA-only planning from secondary bsl-relative integration. The bsl rows are comparison/evaluation rows, not the main MoDA-only planner metric.

|method|top1|role|fixed|harmed|net|
|---|---:|---|---:|---:|---:|
|raw_moda_stateroll_only|55.33|MoDA-only direct planner baseline||||
|best_moda_only_top1::full_bce_pairwise_listwise_preserve|56.33|Best direct MoDA-only calibrated planner by OOF top1||||
|best_moda_only_auc::bce_pairwise_listwise|56.00|Best MoDA-only calibration by candidate AUC; may not maximize direct top1||||
|bsl|81.00|Strong baseline reference only||||
|bsl_integrated_conservative_rank_preserve|82.00|Secondary bsl-relative integration result|6|0|6|
|bsl_integrated_detector_v2_gate_rank_preserve|82.00|Secondary bsl-relative integration result|6|0|6|
|bsl_integrated_selector_v3_balanced|83.17|Secondary bsl-relative integration result|15|2|13|
