#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_l003_l007_resume
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

BASE_CKPT="$STABLE/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4_object.ckpt"
BASE_PREFIX=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07

setting_params() {
  case "$1" in
    standard) echo "300 30 30" ;;
    medium) echo "600 60 20" ;;
    strong) echo "1000 100 20" ;;
    *) echo "bad cem $1" >&2; return 2 ;;
  esac
}
subdir_for() { echo "${BASE_PREFIX}_staterollseq_$1"; }
model_for() { echo "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_$1_epoch_$2"; }
ckpt_for() { echo "$STABLE/$(subdir_for "$1")/$(model_for "$1" "$2")_object.ckpt"; }

train_branch() {
  local gpu=$1 tag=$2 weight=$3
  local subdir model
  subdir=$(subdir_for "$tag")
  model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}"
  if [[ -f "$(ckpt_for "$tag" 2)" ]]; then
    echo "[SKIP_TRAIN] $(date -Is) gpu=$gpu tag=$tag epoch2 exists"
    return 0
  fi
  echo "[TRAIN] $(date -Is) gpu=$gpu tag=$tag lambda=$weight"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train_encoder_moda_rank_full.py \
    --config-name=lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07 \
    subdir="$subdir" \
    output_model_name="$model" \
    warm_start_ckpt="$BASE_CKPT" \
    warm_start_strict=false \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=2 \
    +trainer.limit_train_batches=300 \
    +trainer.limit_val_batches=30 \
    trainer.precision=bf16 \
    wandb.enabled=false \
    loader.batch_size=32 \
    loader.num_workers=4 \
    +train.freeze_encoder=true \
    +train.freeze_projector=true \
    +loss.state_rollout.enabled=true \
    +loss.state_rollout.weight="$weight" \
    +loss.state_rollout.hidden_dim=256 \
    > "$OUTDIR/train_${tag}.log" 2>&1
  echo "[TRAIN_DONE] $(date -Is) gpu=$gpu tag=$tag"
}

eval_one() {
  local gpu=$1 tag=$2 ep=$3 cem=$4 seed=$5
  local params samples topk steps subdir model ckpt out
  params=$(setting_params "$cem")
  read -r samples topk steps <<< "$params"
  subdir=$(subdir_for "$tag")
  model=$(model_for "$tag" "$ep")
  ckpt=$(ckpt_for "$tag" "$ep")
  out="staterollseq_${tag}_ep${ep}_${cem}_seed${seed}_h4_s${samples}_k${topk}_n${steps}.txt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP_MISSING_CKPT] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed"
    return 0
  fi
  if [[ -f "$STABLE/$subdir/$out" ]]; then
    echo "[SKIP_EXISTS] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed"
    return 0
  fi
  echo "[EVAL] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="${subdir}/${model}" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples="$samples" \
    solver.topk="$topk" \
    solver.n_steps="$steps" \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_${tag}_ep${ep}_${cem}_seed${seed}.log" 2>&1
  echo "[EVAL_DONE] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed"
}

collect() {
  "$PY" /data1/jingyixi/wm_runs/state_rollout_collect_l003_l007.py
}

run_branch() {
  local gpu=$1 tag=$2 weight=$3
  train_branch "$gpu" "$tag" "$weight"
  for ep in 1 2; do
    for cem in standard medium strong; do
      for seed in 42 43 44; do
        eval_one "$gpu" "$tag" "$ep" "$cem" "$seed"
      done
    done
  done
}

case "${1:-all}" in
  l003) run_branch 2 l003 0.03 ;;
  l007) run_branch 3 l007 0.07 ;;
  collect) collect ;;
  all)
    echo "[START] $(date -Is)" | tee "$OUTDIR/master.log"
    (run_branch 2 l003 0.03 > "$OUTDIR/l003.master.log" 2>&1; echo l003_done >> "$OUTDIR/master.log") & echo $! > "$OUTDIR/l003.pid"
    (run_branch 3 l007 0.07 > "$OUTDIR/l007.master.log" 2>&1; echo l007_done >> "$OUTDIR/master.log") & echo $! > "$OUTDIR/l007.pid"
    wait
    collect > "$OUTDIR/collect.log" 2>&1 || true
    cat "$OUTDIR/summary.md" 2>/dev/null || true
    ;;
  *) echo "usage: $0 [all|l003|l007|collect]" >&2; exit 2 ;;
esac
