#!/usr/bin/env bash
set -euo pipefail

cd /data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean

mkdir -p /data1/jingyixi/wm_runs/cost_gap/baseline_compare
mkdir -p /data1/jingyixi/wm_runs
mkdir -p /data1/jingyixi/tmp/tmp /data1/jingyixi/tmp/xdg /data1/jingyixi/tmp/triton

export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export PYTHONPATH=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
export STABLEWM_HOME=/data1/jingyixi/.stable_worldmodel
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton

PY=/data1/jingyixi/conda_envs/lewm5090/bin/python

for ep in 13 17 22 27 34; do
  out=/data1/jingyixi/wm_runs/cost_gap/baseline_compare/epoch_${ep}.json
  log=/data1/jingyixi/wm_runs/cost_gap/baseline_compare/epoch_${ep}.log
  if [ -f "$out" ]; then
    echo "SKIP_BASELINE_EP:$ep"
    continue
  fi
  echo "RUN_BASELINE_EP:$ep"
  "$PY" analyze_cem_margin.py \
    --policy "pusht_baseline64_clean_5090_gpu1/lewm_pusht_baseline64_clean_epoch_${ep}" \
    --cache-dir /data1/jingyixi/.stable_worldmodel \
    --output "$out" \
    --num-eval 20 \
    --num-candidates 64 \
    --seed 42 > "$log" 2>&1
  echo "DONE_BASELINE_EP:$ep"
done

TRAIN_LOG=/data1/jingyixi/wm_runs/lewm_encoder_moda_v14_full_visible_bs32_pred6_gpu2.log
nohup bash -lc '
  cd /data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean &&
  export CUDA_VISIBLE_DEVICES=2 &&
  export STABLEWM_HOME=/data1/jingyixi/.stable_worldmodel &&
  export PYTHONPATH=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean &&
  export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton &&
  export TMPDIR=/data1/jingyixi/tmp/tmp &&
  export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg &&
  export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton &&
  /data1/jingyixi/conda_envs/lewm5090/bin/python train_encoder_moda.py --config-name lewm_encoder_moda_v14_full_visible_bs32_pred6
' > "$TRAIN_LOG" 2>&1 < /dev/null &

echo "PRED6_PID:$!"
