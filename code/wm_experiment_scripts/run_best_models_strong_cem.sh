#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/best_models_strong_cem
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

run_one() {
  local gpu=$1 tag=$2 policy=$3
  local out="${tag}_s1000_k100_n30.txt"
  local log="$OUTDIR/${tag}_s1000_k100_n30.log"
  echo "RUN $tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=20 seed=42 \
    solver.num_samples=1000 solver.topk=100 solver.n_steps=30 \
    output.filename="$out" > "$log" 2>&1
  echo "DONE $tag"
}

run_one 0 gate07_ep2 pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_2 &
run_one 1 gate07_ep3 pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_3 &
run_one 2 gate07_ep16 pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_16 &
run_one 3 pred6_ep3 pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_3 &
wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

items = {
    "gate07_ep2": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/gate07_ep2_s1000_k100_n30.txt"),
    "gate07_ep3": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/gate07_ep3_s1000_k100_n30.txt"),
    "gate07_ep16": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/gate07_ep16_s1000_k100_n30.txt"),
    "pred6_ep3": Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6/pred6_ep3_s1000_k100_n30.txt"),
}
rows = []
for tag, path in items.items():
    txt = path.read_text(errors="ignore") if path.exists() else ""
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    rows.append({"tag": tag, "success_rate": float(m.group(1)) if m else None, "file": str(path)})
out = Path("/data1/jingyixi/wm_runs/best_models_strong_cem/summary.json")
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
