#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-/Users/wangyijing/lewm_migration_bundle/wm_runs}"
DST="${2:-/Users/wangyijing/lewm_hf_artifacts_subset/wm_runs}"
mkdir -p "$DST"
include_dirs=(
  stateroll_normalbudget_candidate_pool_s300_steps30_n100
  bsl_normalbudget_candidate_pool_s300_steps30_n100
  stateroll_normalbudget_candidate_pool_s300_steps30_n50
  pac_moda_v2_full_n100_20260529
  pac_moda_v2_gain_boost_20260529
  pac_moda_v2_opportunity_detector_v2_20260529
  pac_moda_v2_budget_generalization_20260529
  cost_calibration_v2_n100_20260529
  rpn_residual_proposal
  opportunity_conditioned_moda_residual_n100_20260528
  pa_moda_action_conditioned_gate
  worldmodel_pga_mixedpool_20260518
  worldmodel_pga_ablation_20260517
  worldmodel_planning_geometry_alignment
  worldmodel_planning_geometry_alignment_smoke_evalmode
)
for d in "${include_dirs[@]}"; do
  if [ -e "$SRC/$d" ]; then
    echo "Copying $d"
    rsync -a --info=progress2 "$SRC/$d" "$DST/"
  else
    echo "Missing $SRC/$d" >&2
  fi
done
