#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/expert_clip_cem_bundle
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

wait_for_behavior() {
  if pgrep -f "run_after_smooth_behavior_cem_bundle.sh" >/dev/null; then
    echo "[WAIT] behavior bundle still running; expert-clip bundle will start after it exits"
  fi
  while pgrep -f "run_after_smooth_behavior_cem_bundle.sh" >/dev/null; do
    sleep 60
  done
  echo "[START] behavior bundle no longer running; launching expert-clip CEM"
}

run_clip() {
  local gpu=$1
  local clip=$2
  local varscale=$3
  local ep=$4
  local seed=$5
  local clip_tag=${clip/./p}
  local var_tag=${varscale/./p}
  local tag="expertclip_c${clip_tag}_v${var_tag}_gate07_ep${ep}_seed${seed}_h4_s1000_k100_n20"
  local policy="${POLICY_ROOT}/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  local log="${OUTDIR}/${tag}.log"
  local out="${tag}.txt"
  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver._target_=cem_expert_clip_solver.ExpertClipCEMSolver \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    solver.var_scale="$varscale" \
    +solver.action_clip_std="$clip" \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$log" 2>&1
  echo "[DONE] ${tag}"
}

wait_for_behavior

# Two pure expert-support clips and two combined with the promising conservative variance.
(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_clip 0 1.5 1.0 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu0_clip15_var10.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_clip 1 2.0 1.0 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu1_clip20_var10.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_clip 2 2.0 0.5 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu2_clip20_var05.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43 44; do
      run_clip 3 2.5 0.5 "$ep" "$seed"
    done
  done
) > "$OUTDIR/gpu3_clip25_var05.log" 2>&1 &

wait
echo "[ALL DONE] expert-clip CEM"
"$PY" "$ROOT/summarize_expert_clip_cem_bundle.py"
