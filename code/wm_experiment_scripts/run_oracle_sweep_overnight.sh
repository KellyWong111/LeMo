#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
RUNS=/data1/jingyixi/wm_runs
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR="$RUNS/oracle_sweep"
mkdir -p "$OUTDIR" "$RUNS/latest_planning"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

POLICY_EP4="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4"
POLICY_EP17="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_17"

run_oracle() {
  local gpu=$1
  local name=$2
  local num_eval=$3
  local topk=$4
  local samples=$5
  local policy=$6
  local json="$OUTDIR/${name}.json"
  local log="$OUTDIR/${name}.log"
  if [[ -f "$json" ]]; then
    echo "SKIP existing oracle $name"
    return 0
  fi
  echo "RUN oracle gpu=$gpu name=$name eval=$num_eval topk=$topk samples=$samples"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" topk_oracle_pilot.py \
    --policy "$policy" \
    --output "$json" \
    --num-eval "$num_eval" \
    --topk "$topk" \
    --num-samples "$samples" \
    --cem-steps 30 \
    --seed 42 > "$log" 2>&1
}

run_trace() {
  local gpu=$1
  local name=$2
  local policy=$3
  local json="$OUTDIR/${name}.json"
  local log="$OUTDIR/${name}.log"
  if [[ -f "$json" ]]; then
    echo "SKIP existing trace $name"
    return 0
  fi
  echo "RUN trace gpu=$gpu name=$name"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" planning_behavior_case_study.py \
    --names "$name" \
    --policies "$policy" \
    --output "$json" \
    --num-eval 20 \
    --num-samples 300 \
    --topk 30 \
    --cem-steps 30 \
    --seed 42 > "$log" 2>&1
}

run_planning() {
  local gpu=$1
  local name=$2
  local policy=$3
  local result_dir="$STABLE/$(dirname "$policy")"
  local out="${name}_seed42_s300_n30_k30.txt"
  local log="$RUNS/latest_planning/${name}_seed42_s300_n30_k30.log"
  if [[ -f "$result_dir/$out" ]]; then
    echo "SKIP existing planning $name"
    return 0
  fi
  echo "RUN planning gpu=$gpu name=$name"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=20 seed=42 \
    solver.num_samples=300 solver.n_steps=30 solver.topk=30 \
    output.filename="$out" > "$log" 2>&1
}

summarize() {
  "$PY" - <<'PY'
import json
import re
from pathlib import Path

outdir = Path("/data1/jingyixi/wm_runs/oracle_sweep")
summary = {"oracle": {}, "trace": {}, "planning": {}}

for path in sorted(outdir.glob("gate07_ep4_e20_top*.json")):
    data = json.load(open(path))
    summary["oracle"][path.stem] = {
        "top1_success_rate": data.get("top1_success_rate"),
        "oracle_topk_success_rate": data.get("oracle_topk_success_rate"),
        "oracle_gap": None
        if data.get("top1_success_rate") is None or data.get("oracle_topk_success_rate") is None
        else data["oracle_topk_success_rate"] - data["top1_success_rate"],
        "first_success_rank": data.get("first_success_rank"),
        "top1_episode_successes": data.get("top1_episode_successes"),
        "oracle_episode_successes": data.get("oracle_episode_successes"),
    }

for path in sorted(outdir.glob("*trace*.json")):
    data = json.load(open(path))
    for name, record in data.get("models", {}).items():
        summary["trace"][name] = record.get("aggregate", {})

planning_dir = Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07")
for path in sorted(planning_dir.glob("gate07_ep17_seed42_s300_n30_k30.txt")):
    txt = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    summary["planning"][path.stem] = float(m.group(1)) if m else None

(outdir / "summary.json").write_text(json.dumps(summary, indent=2))
lines = [json.dumps(summary, indent=2)]
(outdir / "summary.txt").write_text("\n".join(lines))
print(json.dumps(summary, indent=2))
PY
}

echo "==== overnight oracle sweep started $(date) ===="

run_oracle 1 gate07_ep4_e20_top30_s300 20 30 300 "$POLICY_EP4" &
pid_top30=$!

run_oracle 3 gate07_ep4_e20_top60_s600 20 60 600 "$POLICY_EP4" &
pid_top60=$!

(
  while pgrep -f "topk_oracle_gate07_ep4_s42_e10_top60_n600" >/dev/null; do
    echo "WAIT current e10/top60/s600 job before e20/top100/s1000 $(date)"
    sleep 120
  done
  run_oracle 0 gate07_ep4_e20_top100_s1000 20 100 1000 "$POLICY_EP4"
) &
pid_top100=$!

(
  run_planning 1 gate07_ep17 "$POLICY_EP17"
  run_trace 1 gate07_ep17_trace "$POLICY_EP17"
) &
pid_ep17=$!

status=0
for pid in "$pid_top30" "$pid_top60" "$pid_top100" "$pid_ep17"; do
  if ! wait "$pid"; then
    status=1
    echo "WARN: job pid=$pid failed"
  fi
done

summarize || status=1

echo "==== overnight oracle sweep finished $(date) status=$status ===="
exit "$status"
