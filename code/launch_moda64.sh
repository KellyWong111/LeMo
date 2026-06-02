#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-5}"
EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-lewm_moda64}"
SUBDIR="${SUBDIR:-pusht_moda64}"

cd /home/internship/wm_transfer_lab/LeWM_src/le-wm-main
source /home/internship/miniconda3/etc/profile.d/conda.sh
conda activate lewm

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[MoDA64] gpu=${GPU_ID} epochs=${EPOCHS} run=${RUN_NAME}"
python train_moda.py \
  data=pusht \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.max_epochs="${EPOCHS}" \
  output_model_name="${RUN_NAME}" \
  subdir="${SUBDIR}" \
  wandb.enabled=False
