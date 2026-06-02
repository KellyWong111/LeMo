# Opportunity-Conditioned MoDA Residual Alignment n100

|split|epoch|val direct|direct harm|gate thr|val gated|gated harm|switches|st-only recovered|st top1 before->after|st-only rank before->after|st-only top1/3/5|gate success/fail|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
|splitA_train42_44_val45_47|30|58.3|83|10.00|71.7|34|88|3/19|55.0->47.3|6.1->4.8|8/13/13|0.0177/0.0177|
|splitB_train45_47_val42_44|30|57.3|79|10.00|73.7|19|47|0/18|55.7->44.0|3.9->4.6|6/12/14|0.0176/0.0176|

OOF gated: bsl 81.0 -> selector 72.7, union oracle 94.7, fixed=3, harmed=53, switches=135, stateroll-only recovered=3/37
