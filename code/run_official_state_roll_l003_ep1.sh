#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/official_state_roll_l003_ep1
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

SUBDIR=pusht_official_clean_stateroll_l003_ep1
MODEL_NAME=lewm_pusht_official_clean_stateroll_l003
WARM=/data1/jingyixi/.stable_worldmodel/pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13_object.ckpt
CKPT="$STABLE/$SUBDIR/${MODEL_NAME}_epoch_1_object.ckpt"
POLICY="$SUBDIR/${MODEL_NAME}_epoch_1"

collect() {
  "$PY" /data1/jingyixi/wm_runs/official_state_roll_l003_ep1/official_state_roll_collect.py
}

train_one_epoch() {
  if [[ -s "$CKPT" ]]; then
    echo "[SKIP TRAIN] existing $CKPT" | tee -a "$OUTDIR/master.log"
    return 0
  fi
  echo "[TRAIN START] $(date -Is)" | tee -a "$OUTDIR/master.log"
  CUDA_VISIBLE_DEVICES=2 "$PY" train_official_state_roll.py --config-name=lewm \
    subdir="$SUBDIR" \
    output_model_name="$MODEL_NAME" \
    +warm_start_ckpt="$WARM" \
    +warm_start_strict=false \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=1 \
    trainer.precision=bf16 \
    wandb.enabled=false \
    loader.batch_size=32 \
    loader.num_workers=4 \
    +loss.state_rollout.enabled=true \
    +loss.state_rollout.weight=0.03 \
    +loss.state_rollout.hidden_dim=256 \
    > "$OUTDIR/train.log" 2>&1
  echo "[TRAIN DONE] $(date -Is)" | tee -a "$OUTDIR/master.log"
  ls -lh "$CKPT" | tee -a "$OUTDIR/master.log"
}

run_eval() {
  local gpu=$1
  local seed=$2
  local out="official_stateroll_l003_ep1_seed${seed}_h4_s300_k30_n30.txt"
  local result_path="$STABLE/$SUBDIR/$out"
  if [[ -s "$result_path" ]] && grep -q "success_rate" "$result_path"; then
    echo "[SKIP EVAL] seed=$seed existing $result_path" | tee "$OUTDIR/seed${seed}.status"
    return 0
  fi
  echo "[EVAL START] $(date -Is) gpu=$gpu seed=$seed" | tee "$OUTDIR/seed${seed}.status"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$POLICY" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples=300 \
    solver.topk=30 \
    solver.n_steps=30 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" \
    > "$OUTDIR/eval_seed${seed}.log" 2>&1
  echo "[EVAL DONE] $(date -Is) gpu=$gpu seed=$seed" | tee -a "$OUTDIR/seed${seed}.status"
}

case "${1:-all}" in
  train)
    train_one_epoch
    ;;
  eval)
    for seed in 42 44 46; do run_eval 2 "$seed"; collect || true; done &
    p1=$!
    for seed in 43 45 47; do run_eval 3 "$seed"; collect || true; done &
    p2=$!
    wait "$p1" "$p2"
    collect
    ;;
  collect)
    collect
    ;;
  all)
    echo "[START] $(date -Is)" | tee -a "$OUTDIR/master.log"
    train_one_epoch
    bash "$0" eval
    echo "[DONE] $(date -Is)" | tee -a "$OUTDIR/master.log"
    ;;
  *)
    echo "usage: $0 {all|train|eval|collect}" >&2
    exit 2
    ;;
esac
