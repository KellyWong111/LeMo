#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_overnight_confirm
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
BASE_PREFIX=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07

setting_params() {
  case "$1" in
    standard) echo "300 30 30" ;;
    medium) echo "600 60 20" ;;
    strong) echo "1000 100 20" ;;
    *) echo "unknown setting $1" >&2; return 2 ;;
  esac
}

subdir_for() { echo "${BASE_PREFIX}_staterollseq_$1"; }
model_for() { echo "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_$1_epoch_$2"; }
ckpt_for() { echo "$STABLE/$(subdir_for "$1")/$(model_for "$1" "$2")_object.ckpt"; }

train_branch() {
  local gpu=$1 tag=$2 weight=$3
  local subdir model log
  subdir=$(subdir_for "$tag")
  model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}"
  log="$OUTDIR/train_${tag}.log"
  if [[ -f "$(ckpt_for "$tag" 2)" ]]; then
    echo "[SKIP_TRAIN] $(date -Is) tag=$tag epoch2 exists" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
    return 0
  fi
  echo "[TRAIN] $(date -Is) gpu=$gpu tag=$tag lambda=$weight" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
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
    > "$log" 2>&1
  echo "[TRAIN_DONE] $(date -Is) gpu=$gpu tag=$tag" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
}

wait_ckpt() {
  local tag=$1 ep=$2 timeout_min=${3:-720}
  local ckpt elapsed=0
  ckpt=$(ckpt_for "$tag" "$ep")
  while [[ ! -f "$ckpt" && $elapsed -lt $timeout_min ]]; do
    sleep 60
    elapsed=$((elapsed + 1))
  done
  [[ -f "$ckpt" ]]
}

eval_one() {
  local gpu=$1 tag=$2 ep=$3 cem=$4 seed=$5
  local params samples topk steps subdir model ckpt out log
  params=$(setting_params "$cem")
  read -r samples topk steps <<< "$params"
  subdir=$(subdir_for "$tag")
  model=$(model_for "$tag" "$ep")
  ckpt=$(ckpt_for "$tag" "$ep")
  out="staterollseq_${tag}_ep${ep}_${cem}_seed${seed}_h4_s${samples}_k${topk}_n${steps}.txt"
  log="$OUTDIR/eval_${tag}_ep${ep}_${cem}_seed${seed}.log"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP_EVAL_MISSING_CKPT] $(date -Is) tag=$tag ep=$ep cem=$cem seed=$seed ckpt=$ckpt" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
    return 0
  fi
  if [[ -f "$STABLE/$subdir/$out" ]]; then
    echo "[SKIP_EVAL_EXISTS] $(date -Is) tag=$tag ep=$ep cem=$cem seed=$seed" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
    return 0
  fi
  echo "[EVAL] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed samples=$samples topk=$topk steps=$steps" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
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
    output.filename="$out" > "$log" 2>&1
  echo "[EVAL_DONE] $(date -Is) gpu=$gpu tag=$tag ep=$ep cem=$cem seed=$seed" | tee -a "$OUTDIR/worker_gpu${gpu}.log"
}

collect() {
  "$PY" - <<'PY'
import json, re
from pathlib import Path
from statistics import mean

stable = Path('/data1/jingyixi/.stable_worldmodel')
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_overnight_confirm')
base = 'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_'
tags = ['l003', 'l005', 'l007', 'l010']
epochs = [1, 2]
cems = {
    'standard': (300, 30, 30),
    'medium': (600, 60, 20),
    'strong': (1000, 100, 20),
}
seeds = [42, 43, 44]

def parse_success(path):
    if not path.exists():
        return None
    text = path.read_text(errors='ignore')
    match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None

raw = []
for tag in tags:
    subdir = stable / f'{base}{tag}'
    for ep in epochs:
        for cem, (samples, topk, steps) in cems.items():
            for seed in seeds:
                candidates = [
                    subdir / f'staterollseq_{tag}_ep{ep}_{cem}_seed{seed}_h4_s{samples}_k{topk}_n{steps}.txt',
                    subdir / f'staterollseq_{tag}_ep{ep}_seed{seed}_h4_s{samples}_k{topk}_n{steps}.txt',
                ]
                value = None
                file = None
                for path in candidates:
                    value = parse_success(path)
                    if value is not None:
                        file = str(path)
                        break
                raw.append({'tag': tag, 'epoch': ep, 'cem': cem, 'seed': seed, 'success_rate': value, 'file': file})

(outdir / 'raw_results.json').write_text(json.dumps(raw, indent=2))
rows = []
for tag in tags:
    for ep in epochs:
        for cem in cems:
            vals_by_seed = {r['seed']: r['success_rate'] for r in raw if r['tag'] == tag and r['epoch'] == ep and r['cem'] == cem}
            vals = [v for v in vals_by_seed.values() if v is not None]
            if not vals:
                continue
            rows.append({
                'setting': f'{tag} ep{ep} {cem}',
                'tag': tag,
                'epoch': ep,
                'cem': cem,
                'seed42': vals_by_seed.get(42),
                'seed43': vals_by_seed.get(43),
                'seed44': vals_by_seed.get(44),
                'mean': mean(vals),
                'min': min(vals),
                'max': max(vals),
                'n': len(vals),
            })
rows.sort(key=lambda r: (r['tag'], r['epoch'], {'standard':0,'medium':1,'strong':2}[r['cem']]))
(outdir / 'aggregate.json').write_text(json.dumps(rows, indent=2))
complete = [r for r in rows if r['n'] == 3]
best_mean = max(complete or rows, key=lambda r: r['mean'], default=None)
best_min = max(complete or rows, key=lambda r: r['min'], default=None)

def fmt(x):
    return 'NA' if x is None else f'{x:.1f}'
lines = ['|setting|seed42|seed43|seed44|mean|min|max|n|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['setting']}|{fmt(r['seed42'])}|{fmt(r['seed43'])}|{fmt(r['seed44'])}|{fmt(r['mean'])}|{fmt(r['min'])}|{fmt(r['max'])}|{r['n']}|")
lines.append('')
lines.append('|criterion|setting|mean|min|max|n|')
lines.append('|---|---|---:|---:|---:|---:|')
if best_mean:
    lines.append(f"|highest mean|{best_mean['setting']}|{fmt(best_mean['mean'])}|{fmt(best_mean['min'])}|{fmt(best_mean['max'])}|{best_mean['n']}|")
if best_min:
    lines.append(f"|highest min|{best_min['setting']}|{fmt(best_min['mean'])}|{fmt(best_min['min'])}|{fmt(best_min['max'])}|{best_min['n']}|")
for cem in ['standard', 'medium', 'strong']:
    candidates = [r for r in complete if r['cem'] == cem]
    if candidates:
        b = max(candidates, key=lambda r: (r['mean'], r['min']))
        lines.append(f"|best {cem}|{b['setting']}|{fmt(b['mean'])}|{fmt(b['min'])}|{fmt(b['max'])}|{b['n']}|")
(outdir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
}

worker_gpu1() {
  train_branch 1 l003 0.03
  train_branch 1 l007 0.07
  for tag in l003 l007; do
    for ep in 1 2; do
      wait_ckpt "$tag" "$ep" 1 || true
      for cem in standard medium strong; do
        for seed in 42 43 44; do eval_one 1 "$tag" "$ep" "$cem" "$seed"; done
      done
    done
  done
}

worker_gpu2() {
  for seed in 43 44; do eval_one 2 l010 1 standard "$seed"; done
  for seed in 43 44; do eval_one 2 l010 1 strong "$seed"; done
  for tag in l005 l010; do
    for seed in 42 43 44; do eval_one 2 "$tag" 1 medium "$seed"; done
  done
  for tag in l003 l007; do
    wait_ckpt "$tag" 1 720 || true
    for seed in 42 43 44; do eval_one 2 "$tag" 1 medium "$seed"; done
  done
}

worker_gpu3() {
  for tag in l005 l010; do
    for cem in standard strong; do
      for seed in 42 43 44; do eval_one 3 "$tag" 1 "$cem" "$seed"; done
    done
  done
  for tag in l003 l007; do
    wait_ckpt "$tag" 1 720 || true
    for cem in standard strong; do
      for seed in 42 43 44; do eval_one 3 "$tag" 1 "$cem" "$seed"; done
    done
  done
}

case "${1:-all}" in
  gpu1) worker_gpu1 ;;
  gpu2) worker_gpu2 ;;
  gpu3) worker_gpu3 ;;
  collect) collect ;;
  all)
    echo "[START_ALL] $(date -Is)" | tee "$OUTDIR/master.log"
    (worker_gpu1 > "$OUTDIR/gpu1.master.log" 2>&1; echo gpu1_done >> "$OUTDIR/master.log") & echo $! > "$OUTDIR/gpu1.pid"
    (worker_gpu2 > "$OUTDIR/gpu2.master.log" 2>&1; echo gpu2_done >> "$OUTDIR/master.log") & echo $! > "$OUTDIR/gpu2.pid"
    (worker_gpu3 > "$OUTDIR/gpu3.master.log" 2>&1; echo gpu3_done >> "$OUTDIR/master.log") & echo $! > "$OUTDIR/gpu3.pid"
    wait
    collect > "$OUTDIR/collect.log" 2>&1
    cat "$OUTDIR/summary.md"
    ;;
  *) echo "usage: $0 [all|gpu1|gpu2|gpu3|collect]" >&2; exit 2 ;;
esac
