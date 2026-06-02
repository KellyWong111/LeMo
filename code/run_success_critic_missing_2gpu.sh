#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/success_critic_cem_missing
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
  local w=2.0
  local tag="w${w}_seed${seed}_rerun"
  local out="criticcem_w${w}_seed${seed}_ep9_h4_s1000_k100_n20.txt"
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

run_one 2 42 &
run_one 3 43 &
wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07')
outdir = Path('/data1/jingyixi/wm_runs/success_critic_cem_missing')
rows = []
for p in stable.glob('criticcem_w2.0_seed*_ep9_h4_s1000_k100_n20.txt'):
    text = p.read_text(errors='ignore')
    match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
    seed_match = re.search(r'_seed(\d+)_', p.name)
    if seed_match:
        rows.append({
            'seed': int(seed_match.group(1)),
            'success_rate': float(match.group(1)) if match else None,
            'file': str(p),
        })
rows.sort(key=lambda row: row['seed'])
outdir.joinpath('summary.json').write_text(json.dumps(rows, indent=2))
lines = ['|critic_weight|seed|success_rate|', '|---:|---:|---:|']
for row in rows:
    val = 'NA' if row['success_rate'] is None else f"{row['success_rate']:.1f}"
    lines.append(f"|2.0|{row['seed']}|{val}|")
vals = [row['success_rate'] for row in rows if row['success_rate'] is not None]
if vals:
    lines += ['', f"mean={sum(vals)/len(vals):.1f}, n={len(vals)}"]
outdir.joinpath('summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
