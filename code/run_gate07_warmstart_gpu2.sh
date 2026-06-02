#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="5090-4card"
REMOTE_REPO="/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean"
REMOTE_PY="/data1/jingyixi/conda_envs/lewm5090/bin/python"
REMOTE_LOG="/data1/jingyixi/wm_runs/lewm_encoder_moda_v14_full_visible_bs32_gate07_gpu2.log"

scp \
  /Users/wangyijing/Desktop/le-wm-official-clean/train_encoder_moda.py \
  /Users/wangyijing/Desktop/le-wm-official-clean/utils.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/"

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_REPO}/config/train"
scp \
  /Users/wangyijing/Desktop/le-wm-official-clean/config/train/lewm_encoder_moda_v14_full_visible_bs32_gate07.yaml \
  "${REMOTE_HOST}:${REMOTE_REPO}/config/train/"

ssh "${REMOTE_HOST}" "cd ${REMOTE_REPO} && \
  TMPDIR=/data1/jingyixi/tmp/tmp \
  XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg \
  TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton \
  PYTHONPATH=${REMOTE_REPO} \
  STABLEWM_HOME=/data1/jingyixi/.stable_worldmodel \
  MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton \
  CUDA_VISIBLE_DEVICES=2 \
  nohup ${REMOTE_PY} train_encoder_moda.py --config-name lewm_encoder_moda_v14_full_visible_bs32_gate07 > ${REMOTE_LOG} 2>&1 < /dev/null & echo \$!"
