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
mkdir -p "$RUNS/latest_planning" "$RUNS/cost_gap" "$RUNS/analysis_logs"
cd "$ROOT"

launch_eval() {
  local gpu=$1 name=$2 policy=$3 seed=$4
  SKIPPED_LAST=0
  local out="${name}_seed${seed}_s300_n30_k30.txt"
  local log="$RUNS/latest_planning/${name}_seed${seed}_s300_n30_k30.log"
  local result_dir="$STABLE/$(dirname "$policy")"
  if [[ "$seed" == "42" ]]; then
    case "$name" in
      pred6_ep4) old_out="pred6_ep4_s300_n30_k30.txt" ;;
      pred6_ep7) old_out="pred6_ep7_s300_n30_k30.txt" ;;
      pred6_ep10) old_out="pred6_ep10_s300_n30_k30.txt" ;;
      gate07_ep1) old_out="pred6_gate07_ep1_s300_n30_k30.txt" ;;
      gate07_ep4) old_out="pred6_gate07_ep4_s300_n30_k30.txt" ;;
      gate07_ep7) old_out="pred6_gate07_ep7_s300_n30_k30.txt" ;;
      *) old_out="" ;;
    esac
    if [[ -n "${old_out:-}" && -f "$result_dir/$old_out" ]]; then
      echo "SKIP existing historical planning $old_out"
      SKIPPED_LAST=1
      return 0
    fi
  fi
  if [[ -f "$result_dir/$out" ]]; then
    echo "SKIP existing planning $out"
    SKIPPED_LAST=1
    return 0
  fi
  echo "LAUNCH planning gpu=$gpu name=$name seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=20 seed="$seed" \
    solver.num_samples=300 solver.n_steps=30 solver.topk=30 \
    output.filename="$out" > "$log" 2>&1 < /dev/null &
  echo "$! $name seed=$seed gpu=$gpu" >> "$RUNS/analysis_logs/launched_planning.pids"
}

# Existing aligned checkpoint set: gate07 is warm-started from pred6 epoch_3.
declare -A POLICIES
POLICIES[pred6_ep4]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_4'
POLICIES[pred6_ep7]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7'
POLICIES[pred6_ep10]='pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_10'
POLICIES[gate07_ep1]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_1'
POLICIES[gate07_ep4]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4'
POLICIES[gate07_ep7]='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_7'

# We already have seed42 for all six, but launching is idempotent by output file.
seeds=(42 43 44)
gpu=0
running=0
: > "$RUNS/analysis_logs/launched_planning.pids"
for seed in "${seeds[@]}"; do
  for name in pred6_ep4 pred6_ep7 pred6_ep10 gate07_ep1 gate07_ep4 gate07_ep7; do
    launch_eval $gpu "$name" "${POLICIES[$name]}" "$seed"
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

echo "Launched planning jobs. Monitor with:"
echo "  ps -u jingyixi -o pid,stat,etime,cmd | grep eval.py"
echo "  tail -f $RUNS/latest_planning/*.log"
