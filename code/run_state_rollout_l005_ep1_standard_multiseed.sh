#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_l005_ep1_standard_multiseed
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

run_eval() {
  local gpu=$1
  local seed=$2
  local tag=l005
  local subdir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_${tag}_epoch_1"
  local out="staterollseq_${tag}_ep1_seed${seed}_h4_s300_k30_n30.txt"
  echo "[START] $(date -Is) gpu=$gpu seed=$seed" | tee "$OUTDIR/seed${seed}.status"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="${subdir}/${model}" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples=300 \
    solver.topk=30 \
    solver.n_steps=30 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_seed${seed}.log" 2>&1
  echo "[DONE] $(date -Is) gpu=$gpu seed=$seed" | tee -a "$OUTDIR/seed${seed}.status"
}

run_eval 0 42 &
run_eval 1 43 &
run_eval 2 44 &
wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l005')
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_l005_ep1_standard_multiseed')
rows = []
for p in stable.glob('staterollseq_l005_ep1_seed*_h4_s300_k30_n30.txt'):
    text = p.read_text(errors='ignore')
    match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
    seed = int(re.search(r'_seed(\d+)_', p.name).group(1))
    rows.append({'seed': seed, 'success_rate': float(match.group(1)) if match else None, 'file': str(p)})
rows.sort(key=lambda r: r['seed'])
(outdir / 'summary.json').write_text(json.dumps(rows, indent=2))
vals = [r['success_rate'] for r in rows if r['success_rate'] is not None]
lines = ['|seed|success_rate|', '|---:|---:|']
for r in rows:
    val = 'NA' if r['success_rate'] is None else f"{r['success_rate']:.1f}"
    lines.append(f"|{r['seed']}|{val}|")
if vals:
    lines += ['', f"mean={sum(vals)/len(vals):.1f}, min={min(vals):.1f}, max={max(vals):.1f}, n={len(vals)}"]
(outdir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
