#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
RUNS=/data1/jingyixi/wm_runs
STABLE=/data1/jingyixi/.stable_worldmodel
LOGDIR="$RUNS/latest_planning_new"
mkdir -p "$LOGDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

run_eval() {
  local gpu=$1
  local family=$2
  local epoch=$3
  local policy_dir policy_prefix
  if [[ "$family" == "pred6" ]]; then
    policy_dir="pusht_encoder_moda_v14_full_visible_bs32_pred6"
    policy_prefix="lewm_encoder_moda_v14_full_visible_bs32_pred6"
  else
    policy_dir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
    policy_prefix="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07"
  fi

  local ckpt="$STABLE/$policy_dir/${policy_prefix}_epoch_${epoch}_object.ckpt"
  local policy="$policy_dir/${policy_prefix}_epoch_${epoch}"
  local out="${family}_ep${epoch}_seed42_s300_n30_k30.txt"
  local result="$STABLE/$policy_dir/$out"
  local log="$LOGDIR/${family}_ep${epoch}_seed42_s300_n30_k30.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP missing ckpt $family ep$epoch"
    return 0
  fi
  if [[ -f "$result" ]]; then
    echo "SKIP existing planning $out"
    return 0
  fi

  echo "RUN planning gpu=$gpu family=$family epoch=$epoch"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=20 seed=42 \
    solver.num_samples=300 solver.n_steps=30 solver.topk=30 \
    output.filename="$out" > "$log" 2>&1
  echo "DONE planning family=$family epoch=$epoch"
}

run_queue() {
  local gpu=$1
  shift
  for item in "$@"; do
    run_eval "$gpu" "${item%%:*}" "${item##*:}"
  done
}

summarize() {
  "$PY" - <<'PY'
import json
import re
from pathlib import Path

roots = {
    "pred6": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6"),
    "gate07": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"),
}
summary = {}
for family, root in roots.items():
    rows = []
    for path in sorted(root.glob(f"{family}_ep*_seed42_s300_n30_k30.txt")):
        m_epoch = re.search(r"_ep(\d+)_", path.name)
        txt = path.read_text(errors="ignore")
        m = re.search(r"success_rate':\s*([0-9.]+)", txt)
        rows.append({
            "epoch": int(m_epoch.group(1)) if m_epoch else None,
            "file": str(path),
            "success_rate": float(m.group(1)) if m else None,
        })
    summary[family] = sorted(rows, key=lambda x: x["epoch"] if x["epoch"] is not None else 999)

out = Path("/data1/jingyixi/wm_runs/latest_planning_new/summary_seed42.json")
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
}

gpu1_items=(
  pred6:1 pred6:2 pred6:3 pred6:5 pred6:6 pred6:8 pred6:9 pred6:11 pred6:12 pred6:13
  pred6:14 pred6:15 pred6:16 pred6:17 pred6:18 pred6:19 pred6:20
)
gpu2_items=(
  gate07:2 gate07:3 gate07:5 gate07:6 gate07:8 gate07:9 gate07:10 gate07:11 gate07:12
  gate07:13 gate07:14 gate07:15 gate07:16 gate07:17
)

echo "==== new pred6/gate07 planning started $(date) ===="
run_queue 1 "${gpu1_items[@]}" &
pid1=$!
run_queue 2 "${gpu2_items[@]}" &
pid2=$!

status=0
wait "$pid1" || status=1
wait "$pid2" || status=1
summarize || status=1
echo "==== new pred6/gate07 planning finished $(date) status=$status ===="
exit "$status"
