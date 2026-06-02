#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_alignment
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

BASE_CKPT="$STABLE/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4_object.ckpt"

train_one() {
  local gpu=$1
  local tag=$2
  local weight=$3
  local subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}"
  echo "[TRAIN] $(date -Is) gpu=$gpu tag=$tag lambda=$weight"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train_encoder_moda_rank_full.py \
    --config-name=lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07 \
    subdir="$subdir" \
    output_model_name="$model" \
    warm_start_ckpt="$BASE_CKPT" \
    warm_start_strict=false \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=4 \
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
  echo "[TRAIN_DONE] $(date -Is) tag=$tag"
}

eval_one() {
  local gpu=$1
  local tag=$2
  local ep=$3
  local subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}_epoch_${ep}"
  local ckpt="$STABLE/$subdir/${model}_object.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP_EVAL] missing $ckpt"
    return 0
  fi
  local out="stateroll_${tag}_ep${ep}_seed42_h4_s1000_k100_n20.txt"
  echo "[EVAL] $(date -Is) gpu=$gpu tag=$tag ep=$ep"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="${subdir}/${model}" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed=42 \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_${tag}_ep${ep}.log" 2>&1
  echo "[EVAL_DONE] $(date -Is) tag=$tag ep=$ep"
}

probe_one() {
  local gpu=$1
  local tag=$2
  local ep=$3
  local subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_${tag}_epoch_${ep}"
  local ckpt="$STABLE/$subdir/${model}_object.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP_PROBE] missing $ckpt"
    return 0
  fi
  echo "[PROBE] $(date -Is) gpu=$gpu tag=$tag ep=$ep"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" wm_experiment_scripts/state_head_task_cost_probe.py \
    --policy "${subdir}/${model}" \
    --output "$OUTDIR/probe_${tag}_ep${ep}_seed42.json" \
    --cache-dir "$STABLE" \
    --use-model-state-head \
    --eval-seeds 42 \
    --num-eval 10 \
    --topk 30 \
    --num-samples 300 \
    --cem-steps 30 \
    --restarts 1 \
    --beta-grid 0.25,0.5,1.0,2.0 \
    > "$OUTDIR/probe_${tag}_ep${ep}.log" 2>&1
  echo "[PROBE_DONE] $(date -Is) tag=$tag ep=$ep"
}

(
  train_one 2 l005 0.05
  for ep in 1 2 4; do eval_one 2 l005 "$ep"; done
  for ep in 1 2 4; do probe_one 2 l005 "$ep"; done
) > "$OUTDIR/gpu2_l005.master.log" 2>&1 &

(
  train_one 3 l010 0.10
  for ep in 1 2 4; do eval_one 3 l010 "$ep"; done
  for ep in 1 2 4; do probe_one 3 l010 "$ep"; done
) > "$OUTDIR/gpu3_l010.master.log" 2>&1 &

wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel')
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_alignment')
rows = []
for tag in ['l005', 'l010']:
    subdir = stable / f'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_stateroll_{tag}'
    for ep in [1, 2, 4]:
        eval_file = subdir / f'stateroll_{tag}_ep{ep}_seed42_h4_s1000_k100_n20.txt'
        success = None
        if eval_file.exists():
            text = eval_file.read_text(errors='ignore')
            match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
            success = float(match.group(1)) if match else None
        probe_file = outdir / f'probe_{tag}_ep{ep}_seed42.json'
        probe = {}
        if probe_file.exists():
            data = json.loads(probe_file.read_text())
            probe = data.get('means', {})
        rows.append({
            'tag': tag,
            'epoch': ep,
            'success_rate': success,
            'probe_latent_top1': probe.get('latent_top1'),
            'probe_state_top1': probe.get('state_top1'),
            'probe_oracle': probe.get('oracle'),
        })
rows.sort(key=lambda r: (r['tag'], r['epoch']))
(outdir / 'summary.json').write_text(json.dumps(rows, indent=2))
lines = ['|tag|epoch|standard_success|probe_latent|probe_state|probe_oracle|', '|---|---:|---:|---:|---:|---:|']
for row in rows:
    def fmt(x):
        return 'NA' if x is None else f'{x:.1f}'
    lines.append(
        f"|{row['tag']}|{row['epoch']}|{fmt(row['success_rate'])}|"
        f"{fmt(row['probe_latent_top1'])}|{fmt(row['probe_state_top1'])}|{fmt(row['probe_oracle'])}|"
    )
(outdir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
