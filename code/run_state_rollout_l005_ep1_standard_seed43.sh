#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_l005_ep1_standard_seed43
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

CUDA_VISIBLE_DEVICES=3 "$PY" eval.py --config-name=pusht.yaml \
  policy=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l005/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l005_epoch_1 \
  cache_dir="$STABLE" \
  eval.num_eval=20 \
  seed=43 \
  solver.num_samples=300 \
  solver.topk=30 \
  solver.n_steps=30 \
  plan_config.horizon=4 \
  plan_config.action_block=5 \
  plan_config.receding_horizon=4 \
  output.filename=staterollseq_l005_ep1_seed43_h4_s300_k30_n30.txt \
  > "$OUTDIR/eval_seed43.log" 2>&1
