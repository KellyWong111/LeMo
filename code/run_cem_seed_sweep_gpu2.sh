#!/usr/bin/env bash
set -euo pipefail

cd /data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean

mkdir -p /data1/jingyixi/wm_runs/cem_seed_sweep
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

# Wait for the current GPU2 budget sweep to finish so we do not overlap eval jobs.
while pgrep -f "run_cem_budget_sweep_gpu2.sh|eval.py --config-name=pusht.yaml" >/dev/null; do
  echo "WAIT_EXISTING_EVAL"
  sleep 30
done

run_eval() {
  local tag="$1"
  local policy="$2"
  local cem_seed="$3"
  local out="/data1/jingyixi/wm_runs/cem_seed_sweep/${tag}_solverseed${cem_seed}.txt"
  local log="/data1/jingyixi/wm_runs/cem_seed_sweep/${tag}_solverseed${cem_seed}.log"

  if [ -f "$out" ]; then
    echo "SKIP:$tag:solverseed${cem_seed}"
    return 0
  fi

  echo "RUN:$tag:solverseed${cem_seed}"
  "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir=/data1/jingyixi/.stable_worldmodel \
    eval.num_eval=20 \
    seed=42 \
    solver.seed="$cem_seed" \
    solver.num_samples=300 \
    solver.n_steps=30 \
    solver.topk=30 \
    output.filename="$(basename "$out")" > "$log" 2>&1
  echo "DONE:$tag:solverseed${cem_seed}"
}

declare -a CASES=(
  "fv_ep4 pusht_encoder_moda_v14_full_visible_bs32/lewm_encoder_moda_v14_full_visible_bs32_epoch_4"
  "fv_ep8 pusht_encoder_moda_v14_full_visible_bs32/lewm_encoder_moda_v14_full_visible_bs32_epoch_8"
  "bsl64_ep13 pusht_baseline64_clean_5090_gpu1/lewm_pusht_baseline64_clean_epoch_13"
)

declare -a SOLVER_SEEDS=(0 1 2 3 4)

for case in "${CASES[@]}"; do
  tag="${case%% *}"
  policy="${case#* }"
  for cem_seed in "${SOLVER_SEEDS[@]}"; do
    run_eval "$tag" "$policy" "$cem_seed"
  done
done

