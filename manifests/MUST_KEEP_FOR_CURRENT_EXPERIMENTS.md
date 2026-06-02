# Must-keep artifacts for the current LeWM + MoDA experiments

This list is scoped to the experiments we are actively reporting / may rerun. It intentionally excludes many older failed selectors and auxiliary critics.

## Already included in this code release

Small but important final residual-proposal results:

```text
artifacts/wm_runs/moda_only_learned_residual_proposal_20260530/
```

Includes:

```text
confirm50_audit/moda_only_residual_confirm50_audit.csv
confirm50_audit/moda_only_residual_confirm50_audit.json
confirm50_audit/moda_only_residual_confirm50_audit.md
confirm50_audit/residual_case_studies.csv
confirm50_audit/residual_eval_indices.csv
confirm50_audit/residual_scale_sensitivity.csv
medium/
confirm50_base3_scale025/
smoke/
```

## Must keep in an artifact repo or local cache

These are needed to rerun current MoDA-only residual proposal / calibration diagnostics:

```text
stateroll_normalbudget_candidate_pool_s300_steps30_n100/
bsl_normalbudget_candidate_pool_s300_steps30_n100/
stateroll_normalbudget_candidate_pool_s300_steps30_n50/
bsl_normalbudget_candidate_pool_s300_steps30_n50/
```

These are useful for current result comparison / reporting:

```text
pac_moda_v2_full_n100_corrected_20260529/
pac_moda_v2_selector_v3_detector_gate_20260529/
pac_moda_native_calibration_report outputs if present
cost_calibration_v2_n100_20260529/
cost_calibration_head_n100_20260529/
```

These are proposal / residual related weights worth preserving:

```text
rpn_residual_proposal/
opportunity_conditioned_moda_residual_n100_20260528/
pa_moda_action_conditioned_gate/
```

These world-model alignment checkpoints are worth preserving for provenance:

```text
worldmodel_pga_mixedpool_20260518/
worldmodel_pga_ablation_20260517/
worldmodel_planning_geometry_alignment/
worldmodel_planning_geometry_alignment_smoke_evalmode/
```

## Can omit from first Hugging Face upload

Large failed/side-path experiment families can be kept locally but do not need to be uploaded first:

```text
hard_opportunity_replacement_critic_*
neural_union_success_critic_*
pairwise_replacement_critic_*
two_stage_cascade_*
sequence_mil_*
env_traj_replacement_*
local_near_miss_cost_adapter_*
```

The code scripts remain in `experiments/`, so those methods are documented even if their old weights are not uploaded.
