#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_task_cost_l003_ep1_overnight
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

POLICY=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1
TARGET_DIR=$STABLE/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003

wait_for_current_jobs() {
  local pattern=$1
  local label=$2
  while pgrep -af "$pattern" >/dev/null; do
    echo "[WAIT] $(date -Is) waiting for $label"
    sleep 120
  done
}

run_eval() {
  local gpu=$1
  local beta=$2
  local seed=$3
  local beta_tag=${beta}
  local out="statetask_l003_ep1_b${beta_tag}_seed${seed}_h4_s300_k30_n30.txt"
  if [[ -f "$TARGET_DIR/$out" ]]; then
    echo "[SKIP] $(date -Is) beta=$beta seed=$seed"
    return 0
  fi
  echo "[EVAL] $(date -Is) gpu=$gpu beta=$beta seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$POLICY" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver._target_=state_task_cost_cem_solver.StateTaskCostCEMSolver \
    +solver.state_cost_weight="$beta" \
    solver.num_samples=300 \
    solver.topk=30 \
    solver.n_steps=30 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_b${beta_tag}_seed${seed}.log" 2>&1
}

collect() {
  "$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path('/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003')
outdir = Path('/data1/jingyixi/wm_runs/state_task_cost_l003_ep1_overnight')
betas = ['0.05', '0.1', '0.15', '0.25', '0.5', '1.0']
seeds = [42, 43, 44, 45, 46, 47]
rows = []
for beta in betas:
    row = {'beta': beta}
    vals = []
    for seed in seeds:
        p = stable / f'statetask_l003_ep1_b{beta}_seed{seed}_h4_s300_k30_n30.txt'
        val = None
        if p.exists():
            text = p.read_text(errors='ignore')
            m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
            if m:
                val = float(m.group(1))
        row[f'seed{seed}'] = val
        if val is not None:
            vals.append(val)
    row['mean'] = sum(vals) / len(vals) if vals else None
    row['min'] = min(vals) if vals else None
    row['max'] = max(vals) if vals else None
    row['n'] = len(vals)
    rows.append(row)

(outdir / 'raw_results.json').write_text(json.dumps(rows, indent=2))

def fmt(x):
    return 'NA' if x is None else f'{x:.1f}'

lines = ['|beta|seed42|seed43|seed44|seed45|seed46|seed47|mean|min|max|n|',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(
        f"|{row['beta']}|{fmt(row['seed42'])}|{fmt(row['seed43'])}|{fmt(row['seed44'])}|"
        f"{fmt(row['seed45'])}|{fmt(row['seed46'])}|{fmt(row['seed47'])}|"
        f"{fmt(row['mean'])}|{fmt(row['min'])}|{fmt(row['max'])}|{row['n']}|"
    )

complete = [r for r in rows if r['n'] >= 3 and r['mean'] is not None]
if complete:
    best_mean = max(complete, key=lambda r: r['mean'])
    best_min = max(complete, key=lambda r: r['min'])
    lines += ['', '|criterion|beta|mean|min|max|n|', '|---|---:|---:|---:|---:|---:|']
    for name, row in [('highest mean', best_mean), ('highest min', best_min)]:
        lines.append(f"|{name}|{row['beta']}|{row['mean']:.1f}|{row['min']:.1f}|{row['max']:.1f}|{row['n']}|")

(outdir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
}

worker_gpu2() {
  wait_for_current_jobs "baselinecmp_pred6_ep7" "pred6 baseline queue"
  for beta in 0.05 0.15 1.0; do
    for seed in 42 43 44 45 46 47; do
      run_eval 2 "$beta" "$seed"
      collect
    done
  done
}

worker_gpu3() {
  wait_for_current_jobs "run_state_task_cost_l003_ep1.sh|StateTaskCostCEMSolver" "current state-task queue"
  for beta in 0.1 0.25 0.5; do
    for seed in 42 43 44 45 46 47; do
      run_eval 3 "$beta" "$seed"
      collect
    done
  done
}

case "${1:-all}" in
  collect) collect ;;
  gpu2) worker_gpu2 ;;
  gpu3) worker_gpu3 ;;
  all)
    echo "[START] $(date -Is)" > "$OUTDIR/master.log"
    (worker_gpu2; collect) > "$OUTDIR/gpu2.master.log" 2>&1 & echo $! > "$OUTDIR/gpu2.pid"
    (worker_gpu3; collect) > "$OUTDIR/gpu3.master.log" 2>&1 & echo $! > "$OUTDIR/gpu3.pid"
    wait
    collect > "$OUTDIR/final_collect.log" 2>&1
    echo "[DONE] $(date -Is)" >> "$OUTDIR/master.log"
    ;;
  *) echo "unknown mode: $1" >&2; exit 2 ;;
esac
