#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/reachability_contrast_train
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

BASE_CKPT="$STABLE/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4_object.ckpt"

run_one() {
  local gpu=$1
  local tag=$2
  local weight=$3
  local temp=$4
  local metric=$5
  local max_epochs=$6
  local log="$OUTDIR/${tag}.log"
  echo "[RUN] gpu=${gpu} tag=${tag} weight=${weight} temp=${temp} metric=${metric}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train_encoder_moda_rank_full.py \
    --config-name=lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07 \
    subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_reach_${tag}" \
    output_model_name="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_reach_${tag}" \
    warm_start_ckpt="$BASE_CKPT" \
    warm_start_strict=false \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs="$max_epochs" \
    trainer.precision=bf16 \
    wandb.enabled=false \
    loader.batch_size=32 \
    loader.num_workers=4 \
    +loss.reachability_contrast.enabled=true \
    +loss.reachability_contrast.weight="$weight" \
    +loss.reachability_contrast.temperature="$temp" \
    +loss.reachability_contrast.metric="$metric" \
    +loss.reachability_contrast.normalize=true \
    > "$log" 2>&1
  echo "[DONE] ${tag}"
}

run_one 0 w003_t010_cos 0.03 0.10 cosine 12 &
pid0=$!
run_one 1 w010_t010_cos 0.10 0.10 cosine 12 &
pid1=$!
run_one 2 w003_t005_cos 0.03 0.05 cosine 12 &
pid2=$!
run_one 3 w010_t005_cos 0.10 0.05 cosine 12 &
pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    status=1
  fi
done
echo "[ALL DONE] status=${status}"
exit "$status"
