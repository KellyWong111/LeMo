#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/cem_hparam_sweep_now
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"
POLICY=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4

run_one() {
  local gpu=$1 samples=$2 topk=$3 steps=$4 tag=$5
  local out="${tag}.txt"
  local log="$OUTDIR/${tag}.log"
  echo "RUN $tag gpu=$gpu samples=$samples topk=$topk steps=$steps"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$POLICY" cache_dir="$STABLE" eval.num_eval=20 seed=42 \
    solver.num_samples=$samples solver.topk=$topk solver.n_steps=$steps \
    output.filename="$out" > "$log" 2>&1
  echo "DONE $tag"
}

run_one 0 600 60 30 gate07_ep4_e20_s600_k60_n30 &
run_one 1 600 100 20 gate07_ep4_e20_s600_k100_n20 &
run_one 2 1000 100 15 gate07_ep4_e20_s1000_k100_n15 &
run_one 3 1000 100 30 gate07_ep4_e20_s1000_k100_n30 &
wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

root = Path("/data1/jingyixi/wm_runs/cem_hparam_sweep_now")
stable = Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07")
rows = []
for tag in [
    "gate07_ep4_e20_s600_k60_n30",
    "gate07_ep4_e20_s600_k100_n20",
    "gate07_ep4_e20_s1000_k100_n15",
    "gate07_ep4_e20_s1000_k100_n30",
]:
    path = stable / f"{tag}.txt"
    txt = path.read_text(errors="ignore") if path.exists() else ""
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    rows.append({"tag": tag, "success_rate": float(m.group(1)) if m else None, "file": str(path)})
(root / "summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
