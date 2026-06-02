# Recommended artifacts for a runnable release

The full local migration bundle is about 113G and includes many intermediate experiments. For a clean Hugging Face artifact repo, prioritize the following paths from:

```text
/Users/wangyijing/lewm_migration_bundle/wm_runs
```

## Candidate pools

Required for most calibration / residual proposal diagnostics:

```text
stateroll_normalbudget_candidate_pool_s300_steps30_n100/
bsl_normalbudget_candidate_pool_s300_steps30_n100/
stateroll_normalbudget_candidate_pool_s300_steps30_n50/
```

## Main PAC-MoDA / integration results

```text
pac_moda_v2_full_n100_20260529/
pac_moda_v2_gain_boost_20260529/
pac_moda_v2_opportunity_detector_v2_20260529/
pac_moda_v2_budget_generalization_20260529/
cost_calibration_v2_n100_20260529/
```

## Residual / proposal-related weights

```text
rpn_residual_proposal/
opportunity_conditioned_moda_residual_n100_20260528/
pa_moda_action_conditioned_gate/
```

## World-model planning alignment checkpoints

```text
worldmodel_pga_mixedpool_20260518/
worldmodel_pga_ablation_20260517/
worldmodel_planning_geometry_alignment/
worldmodel_planning_geometry_alignment_smoke_evalmode/
```

## Known missing or server-only latest result

The final confirm50 audit result may not be present locally. If the server becomes available, sync:

```text
/data1/jingyixi/wm_runs/moda_only_learned_residual_proposal_20260530/confirm50_audit/
```

Expected key files:

```text
moda_only_residual_confirm50_audit.csv
moda_only_residual_confirm50_audit.json
moda_only_residual_confirm50_audit.md
residual_case_studies.csv
residual_eval_indices.csv
residual_scale_sensitivity.csv
```
