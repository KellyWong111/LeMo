#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
RUNS=/data1/jingyixi/wm_runs
STABLE=/data1/jingyixi/.stable_worldmodel
export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
mkdir -p "$RUNS/cost_gap_aligned" "$RUNS/analysis_logs"
cd "$ROOT"

declare -A POLICIES
POLICIES[pred6_ep4]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_4'
POLICIES[pred6_ep7]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7'
POLICIES[pred6_ep10]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_10'
POLICIES[gate07_ep1]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_1'
POLICIES[gate07_ep4]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4'
POLICIES[gate07_ep7]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_7'

launch_margin() {
  local gpu=$1 name=$2 policy=$3 seed=$4
  local out="$RUNS/cost_gap_aligned/${name}_seed${seed}_n20_c64.json"
  local log="$RUNS/analysis_logs/${name}_seed${seed}_n20_c64_margin.log"
  if [[ -f "$out" ]]; then
    echo "SKIP existing margin $out"
    SKIPPED_LAST=1
    return 0
  fi
  SKIPPED_LAST=0
  echo "LAUNCH margin gpu=$gpu name=$name seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" analyze_cem_margin.py \
    --policy "$policy" \
    --cache-dir "$STABLE" \
    --num-eval 20 \
    --num-candidates 64 \
    --seed "$seed" \
    --output "$out" > "$log" 2>&1 < /dev/null &
  echo "$! $name seed=$seed gpu=$gpu" >> "$RUNS/analysis_logs/launched_margin_probe.pids"
}

seeds=(42 43 44)
gpu=0
running=0
: > "$RUNS/analysis_logs/launched_margin_probe.pids"
for seed in "${seeds[@]}"; do
  for name in pred6_ep4 pred6_ep7 pred6_ep10 gate07_ep1 gate07_ep4 gate07_ep7; do
    launch_margin "$gpu" "$name" "${POLICIES[$name]}" "$seed"
    if [[ "${SKIPPED_LAST:-0}" != "1" ]]; then
      running=$((running + 1))
    fi
    gpu=$(( (gpu + 1) % 4 ))
    if (( running >= 4 )); then
      wait
      running=0
    fi
  done
done
wait

echo "Aligned margin probe complete."
