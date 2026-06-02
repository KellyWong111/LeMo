#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_best_multiseed
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/triton
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

run_eval() {
  local gpu=$1
  local tag=$2
  local ep=$3
  local seed=$4
  local subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}_epoch_${ep}"
  local out="staterollseq_${tag}_ep${ep}_seed${seed}_h4_s1000_k100_n20.txt"
  echo "[START] $(date -Is) gpu=$gpu tag=$tag ep=$ep seed=$seed" | tee "$OUTDIR/${tag}_ep${ep}_seed${seed}.status"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="${subdir}/${model}" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_${tag}_ep${ep}_seed${seed}.log" 2>&1
  echo "[DONE] $(date -Is) gpu=$gpu tag=$tag ep=$ep seed=$seed" | tee -a "$OUTDIR/${tag}_ep${ep}_seed${seed}.status"
}

run_eval 0 l005 1 42 &
run_eval 1 l005 1 43 &
run_eval 2 l005 1 44 &
run_eval 3 l010 1 42 &
wait

"$PY" - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel')
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_best_multiseed')
rows = []
for tag in ['l005', 'l010']:
    subdir = stable / f'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_{tag}'
    for p in subdir.glob(f'staterollseq_{tag}_ep1_seed*_h4_s1000_k100_n20.txt'):
        text = p.read_text(errors='ignore')
        match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
        sm = re.search(r'_seed(\d+)_', p.name)
        rows.append({
            'tag': tag,
            'epoch': 1,
            'seed': int(sm.group(1)) if sm else None,
            'success_rate': float(match.group(1)) if match else None,
            'file': str(p),
        })
rows.sort(key=lambda row: (row['tag'], row['seed'] or -1))
(outdir / 'summary.json').write_text(json.dumps(rows, indent=2))
lines = ['|tag|epoch|seed|success_rate|', '|---|---:|---:|---:|']
for row in rows:
    val = 'NA' if row['success_rate'] is None else f"{row['success_rate']:.1f}"
    lines.append(f"|{row['tag']}|{row['epoch']}|{row['seed']}|{val}|")
by_tag = defaultdict(list)
for row in rows:
    if row['success_rate'] is not None:
        by_tag[row['tag']].append(row['success_rate'])
lines += ['', '|tag|n|mean|min|max|', '|---|---:|---:|---:|---:|']
for tag, vals in sorted(by_tag.items()):
    lines.append(f"|{tag}|{len(vals)}|{sum(vals)/len(vals):.1f}|{min(vals):.1f}|{max(vals):.1f}|")
(outdir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
