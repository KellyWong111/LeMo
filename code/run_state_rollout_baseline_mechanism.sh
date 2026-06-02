#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/state_rollout_baseline_mechanism
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
  local gpu=$1 name=$2 policy=$3 seed=$4
  local out="baselinecmp_${name}_seed${seed}_h4_s300_k30_n30.txt"
  if [[ -f "$STABLE/${policy%/*}/$out" ]]; then
    echo "[SKIP_EVAL] $name seed=$seed"
    return 0
  fi
  echo "[EVAL] $(date -Is) gpu=$gpu name=$name seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=20 seed="$seed" \
    solver.num_samples=300 solver.topk=30 solver.n_steps=30 \
    plan_config.horizon=4 plan_config.action_block=5 plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_${name}_seed${seed}.log" 2>&1
}

collect() {
"$PY" - <<'PY'
import json, re
from pathlib import Path
stable = Path('/data1/jingyixi/.stable_worldmodel')
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_baseline_mechanism')
settings = [
 ('original_official_ep13','pusht_official_clean_5090_gpu0','baselinecmp_original_official_ep13_seed{seed}_h4_s300_k30_n30.txt'),
 ('pred6_ep7','pusht_encoder_moda_v14_full_visible_bs32_pred6','baselinecmp_pred6_ep7_seed{seed}_h4_s300_k30_n30.txt'),
 ('gate07_ep4','pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07','gate07_epoch4_standard_seed{seed}_h4_s300_k30_n30.txt'),
 ('state_roll_l003_ep1','pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003','staterollseq_l003_ep1_standard_seed{seed}_h4_s300_k30_n30.txt'),
]
seeds = [42,43,44,45,46,47]
rows=[]
for name, subdir, pattern in settings:
    vals=[]; row={'setting':name}
    for seed in seeds:
        p = stable/subdir/pattern.format(seed=seed)
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
(outdir/'baseline_raw_results.json').write_text(json.dumps(rows, indent=2))
lines=['|setting|seed42|seed43|seed44|seed45|seed46|seed47|mean|min|max|n|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
def fmt(x): return 'NA' if x is None else f'{x:.1f}'
for r in rows:
    lines.append('|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|'.format(r['setting'],*(fmt(r[f'seed{s}']) for s in seeds),fmt(r['mean']),fmt(r['min']),fmt(r['max']),r['n']))
# mechanism aggregate if available
mech_path=outdir/'mechanism_raw.json'
if mech_path.exists():
    data=json.loads(mech_path.read_text())
    keys=['top1_success','oracle_top30','rollout_final_spread_mean','final_pr','participation_ratio','dz_da_ratio_mean','cost_gap','pairwise_success_auc']
    lines += ['','## mechanism gate07 vs state-roll','|policy|metric|mean|min|max|n|','|---|---|---:|---:|---:|---:|']
    for policy in ['gate07_ep4','state_roll_l003_ep1']:
        subset=[d for d in data if d['name']==policy]
        for k in keys:
            vals=[d[k] for d in subset if d.get(k) is not None]
            if vals:
                lines.append(f"|{policy}|{k}|{sum(vals)/len(vals):.3f}|{min(vals):.3f}|{max(vals):.3f}|{len(vals)}|")
(outdir/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
}

baseline_worker() {
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 42
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 43
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 44
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 45
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 46
  run_eval 2 original_official_ep13 pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13 47
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 42
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 43
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 44
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 45
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 46
  run_eval 2 pred6_ep7 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7 47
}

mechanism_worker() {
  CUDA_VISIBLE_DEVICES=3 "$PY" wm_experiment_scripts/state_rollout_mechanism_probe.py \
    --output "$OUTDIR/mechanism_raw.json" --cache-dir "$STABLE" \
    --seeds 42,43,44 --num-eval 20 --topk 30 --num-samples 300 --cem-steps 30 \
    > "$OUTDIR/mechanism.log" 2>&1
}

case "${1:-all}" in
  collect) collect ;;
  baseline) baseline_worker ; collect ;;
  mechanism) mechanism_worker ; collect ;;
  all)
    echo "[START] $(date -Is)" > "$OUTDIR/master.log"
    (baseline_worker; collect) > "$OUTDIR/baseline_worker.log" 2>&1 & echo $! > "$OUTDIR/baseline_worker.pid"
    (mechanism_worker; collect) > "$OUTDIR/mechanism_worker.log" 2>&1 & echo $! > "$OUTDIR/mechanism_worker.pid"
    wait
    collect > "$OUTDIR/final_collect.log" 2>&1
    echo "[DONE] $(date -Is)" >> "$OUTDIR/master.log"
    ;;
  *) echo "unknown mode" >&2; exit 2 ;;
esac
