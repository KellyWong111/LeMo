#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/success_critic_cem_multiseed
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

CRITIC=/data1/jingyixi/wm_runs/success_critic_cem/ep9_s42_43_44_n30_top30_scalar_critic.pt
POLICY=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_9

run_one() {
  local gpu=$1
  local seed=$2
  local w=$3
  local tag="w${w}_seed${seed}"
  local out="criticcem_${tag}_ep9_h4_s1000_k100_n20.txt"
  echo "START $(date -Is) gpu=$gpu seed=$seed w=$w" | tee "$OUTDIR/${tag}.status"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$POLICY" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver._target_=cem_success_critic_solver.SuccessCriticCEMSolver \
    +solver.critic_path="$CRITIC" \
    +solver.critic_weight="$w" \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/${tag}.log" 2>&1
  echo "DONE $(date -Is) gpu=$gpu seed=$seed w=$w" | tee -a "$OUTDIR/${tag}.status"
}

run_one 0 42 2.0 & echo $! > "$OUTDIR/w2.0_seed42.pid"
run_one 1 43 2.0 & echo $! > "$OUTDIR/w2.0_seed43.pid"
run_one 2 44 2.0 & echo $! > "$OUTDIR/w2.0_seed44.pid"
run_one 3 42 1.5 & echo $! > "$OUTDIR/w1.5_seed42.pid"
wait

"$PY" - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07')
outdir = Path('/data1/jingyixi/wm_runs/success_critic_cem_multiseed')
rows = []
for p in stable.glob('criticcem_w*_seed*_ep9_h4_s1000_k100_n20.txt'):
    text = p.read_text(errors='ignore')
    m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
    wm = re.search(r'criticcem_w([^_]+)_seed(\d+)_', p.name)
    if wm:
        rows.append({
            'weight': wm.group(1),
            'seed': int(wm.group(2)),
            'success_rate': float(m.group(1)) if m else None,
            'file': str(p),
        })
rows.sort(key=lambda r: (float(r['weight']), r['seed']))
outdir.joinpath('summary.json').write_text(json.dumps(rows, indent=2))
lines = ['|critic_weight|seed|success_rate|', '|---:|---:|---:|']
for r in rows:
    val = 'NA' if r['success_rate'] is None else f"{r['success_rate']:.1f}"
    lines.append(f"|{r['weight']}|{r['seed']}|{val}|")
by_weight = defaultdict(list)
for r in rows:
    if r['success_rate'] is not None:
        by_weight[r['weight']].append(r['success_rate'])
lines += ['', '|critic_weight|n|mean|', '|---:|---:|---:|']
for w, vals in sorted(by_weight.items(), key=lambda kv: float(kv[0])):
    lines.append(f"|{w}|{len(vals)}|{sum(vals)/len(vals):.1f}|")
outdir.joinpath('summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
