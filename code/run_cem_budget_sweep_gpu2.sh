#!/usr/bin/env bash
set -euo pipefail

cd /data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean

mkdir -p /data1/jingyixi/wm_runs/cem_budget_sweep
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

run_eval() {
  local tag="$1"
  local policy="$2"
  local samples="$3"
  local steps="$4"
  local topk="$5"
  local out="/data1/jingyixi/wm_runs/cem_budget_sweep/${tag}_s${samples}_n${steps}_k${topk}.txt"
  local log="/data1/jingyixi/wm_runs/cem_budget_sweep/${tag}_s${samples}_n${steps}_k${topk}.log"

  if [ -f "$out" ]; then
    echo "SKIP:$tag:s${samples}:n${steps}:k${topk}"
    return 0
  fi

  echo "RUN:$tag:s${samples}:n${steps}:k${topk}"
  "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir=/data1/jingyixi/.stable_worldmodel \
    eval.num_eval=20 \
    solver.num_samples="$samples" \
    solver.n_steps="$steps" \
    solver.topk="$topk" \
    output.filename="$(basename "$out")" > "$log" 2>&1
  echo "DONE:$tag:s${samples}:n${steps}:k${topk}"
}

# Three checkpoints:
# - full-visible high-planning early point
# - full-visible later point with lower planning
# - strong baseline reference
declare -a CASES=(
  "fv_ep4 pusht_encoder_moda_v14_full_visible_bs32/lewm_encoder_moda_v14_full_visible_bs32_epoch_4"
  "fv_ep8 pusht_encoder_moda_v14_full_visible_bs32/lewm_encoder_moda_v14_full_visible_bs32_epoch_8"
  "bsl64_ep13 pusht_baseline64_clean_5090_gpu1/lewm_pusht_baseline64_clean_epoch_13"
)

# Cheap / medium / stronger search budgets.
declare -a BUDGETS=(
  "64 10 8"
  "150 15 15"
  "300 30 30"
)

for case in "${CASES[@]}"; do
  tag="${case%% *}"
  policy="${case#* }"
  for budget in "${BUDGETS[@]}"; do
    read -r samples steps topk <<<"$budget"
    run_eval "$tag" "$policy" "$samples" "$steps" "$topk"
  done
done

