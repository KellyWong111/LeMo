#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_head_task_cost
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

CUDA_VISIBLE_DEVICES=2 "$PY" wm_experiment_scripts/state_head_task_cost_probe.py \
  --policy pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_9 \
  --output "$OUTDIR/gate07_ep9_state_head_task_cost_seed42_43_44.json" \
  --cache-dir "$STABLE" \
  --train-samples 5000 \
  --val-samples 1000 \
  --epochs 80 \
  --batch-size 512 \
  --encode-batch 256 \
  --eval-seeds 42,43,44 \
  --num-eval 20 \
  --topk 30 \
  --num-samples 300 \
  --cem-steps 30 \
  --restarts 1 \
  --beta-grid 0.25,0.5,1.0,2.0,4.0 \
  > "$OUTDIR/gate07_ep9_state_head_task_cost.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path

out = Path('/data1/jingyixi/wm_runs/state_head_task_cost')
data = json.loads((out / 'gate07_ep9_state_head_task_cost_seed42_43_44.json').read_text())
lines = ['|metric|value|', '|---|---:|']
for key, val in data['means'].items():
    lines.append(f'|{key}|{val:.1f}|')
lines.append('')
lines.append('|seed|latent_top1|state_top1|oracle|')
lines.append('|---:|---:|---:|---:|')
for row in data['seeds']:
    lines.append(
        f"|{row['seed']}|{row['latent_top1_success_rate']:.1f}|"
        f"{row['state_top1_success_rate']:.1f}|{row['oracle_topk_success_rate']:.1f}|"
    )
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
