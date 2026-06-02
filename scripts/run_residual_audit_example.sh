#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"

if [ ! -d "$LEWM_WM_RUNS/stateroll_normalbudget_candidate_pool_s300_steps30_n100" ]; then
  echo "Missing candidate pool at $LEWM_WM_RUNS/stateroll_normalbudget_candidate_pool_s300_steps30_n100" >&2
  echo "Copy or symlink artifacts/wm_runs before running full evaluation." >&2
  exit 1
fi

cd "$ROOT_DIR/code"
python "$ROOT_DIR/experiments/moda_only_residual_confirm50_audit.py" \
  --num-samples 150 \
  --cem-steps 15 \
  --raw-topk 10 \
  --eval-topk 10 \
  --base-top 3 \
  --scales 0.1,0.2,0.25,0.3,0.4,0.5 \
  --case-scale 0.25 \
  --cal-lambda 1.0 \
  --outdir "$ROOT_DIR/artifacts/residual_confirm50_audit"
