#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_task_cost_l003_ep1
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
run_one() {
  local beta=$1 seed=$2
  local out="statetask_l003_ep1_b${beta}_seed${seed}_h4_s300_k30_n30.txt"
  echo "[EVAL] $(date -Is) beta=$beta seed=$seed"
  CUDA_VISIBLE_DEVICES=3 "$PY" eval.py --config-name=pusht.yaml \
    policy="$POLICY" cache_dir="$STABLE" eval.num_eval=20 seed="$seed" \
    solver._target_=state_task_cost_cem_solver.StateTaskCostCEMSolver \
    +solver.state_cost_weight="$beta" solver.num_samples=300 solver.topk=30 solver.n_steps=30 \
    plan_config.horizon=4 plan_config.action_block=5 plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_b${beta}_seed${seed}.log" 2>&1
}
for beta in 0.1 0.25 0.5; do
  for seed in 42 43 44; do
    run_one "$beta" "$seed"
  done
done
"$PY" - <<'PY'
import json, re
from pathlib import Path
stable=Path('/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003')
outdir=Path('/data1/jingyixi/wm_runs/state_task_cost_l003_ep1')
rows=[]
for beta in ['0.1','0.25','0.5']:
    vals=[]; row={'beta':beta}
    for seed in [42,43,44]:
        p=stable/f'statetask_l003_ep1_b{beta}_seed{seed}_h4_s300_k30_n30.txt'
        val=None
        if p.exists():
            text=p.read_text(errors='ignore')
            m=re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
            if m: val=float(m.group(1))
        row[f'seed{seed}']=val
        if val is not None: vals.append(val)
    row['mean']=sum(vals)/len(vals) if vals else None
    row['min']=min(vals) if vals else None
    row['max']=max(vals) if vals else None
    row['n']=len(vals)
    rows.append(row)
(outdir/'raw_results.json').write_text(json.dumps(rows, indent=2))
def fmt(x): return 'NA' if x is None else f'{x:.1f}'
lines=['|beta|seed42|seed43|seed44|mean|min|max|n|','|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['beta']}|{fmt(r['seed42'])}|{fmt(r['seed43'])}|{fmt(r['seed44'])}|{fmt(r['mean'])}|{fmt(r['min'])}|{fmt(r['max'])}|{r['n']}|")
(outdir/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
