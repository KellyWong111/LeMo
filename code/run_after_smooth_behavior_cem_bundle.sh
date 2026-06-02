#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/behavior_cem_bundle
POLICY_ROOT=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07

mkdir -p "$OUTDIR"
export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

wait_for_smooth() {
  if pgrep -f "/data1/jingyixi/wm_runs/run_smooth_cem_trials.sh" >/dev/null; then
    echo "[WAIT] smooth CEM still running; behavior bundle will start after it exits"
  fi
  while pgrep -f "/data1/jingyixi/wm_runs/run_smooth_cem_trials.sh" >/dev/null; do
    sleep 60
  done
  echo "[START] smooth CEM no longer running; launching behavior bundle"
}

run_shrink() {
  local gpu=$1
  local alpha=$2
  local ep=$3
  local seed=$4
  local alpha_tag=${alpha/./p}
  local tag="shrink_a${alpha_tag}_gate07_ep${ep}_seed${seed}_h4_s1000_k100_n20"
  local policy="${POLICY_ROOT}/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  local log="${OUTDIR}/${tag}.log"
  local out="${tag}.txt"
  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver._target_=cem_shrink_solver.ShrinkCEMSolver \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    +solver.action_shrink="$alpha" \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$log" 2>&1
  echo "[DONE] ${tag}"
}

run_varscale() {
  local gpu=$1
  local varscale=$2
  local ep=$3
  local seed=$4
  local var_tag=${varscale/./p}
  local tag="varscale_v${var_tag}_gate07_ep${ep}_seed${seed}_h4_s1000_k100_n20"
  local policy="${POLICY_ROOT}/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  local log="${OUTDIR}/${tag}.log"
  local out="${tag}.txt"
  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    solver.var_scale="$varscale" \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$log" 2>&1
  echo "[DONE] ${tag}"
}

wait_for_smooth

# Four queues: shrink is primary, var_scale is the cheap control for conservative search.
(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_shrink 0 0.7 "$ep" "$seed"
    done
  done
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_varscale 0 0.5 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu0_alpha07_var05.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_shrink 1 0.8 "$ep" "$seed"
    done
  done
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_varscale 1 0.7 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu1_alpha08_var07.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_shrink 2 0.9 "$ep" "$seed"
    done
  done
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_varscale 2 0.9 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu2_alpha09_var09.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_shrink 3 1.0 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu3_alpha10_control.log" 2>&1 &

wait
echo "[ALL DONE] behavior bundle"
"$PY" "$ROOT/summarize_behavior_cem_bundle.py"
