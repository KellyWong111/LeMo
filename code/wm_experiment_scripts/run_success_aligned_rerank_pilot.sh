#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/success_aligned_rerank
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

run_one() {
  local gpu=$1
  local ep=$2
  local seed=$3
  local tag="gate07_ep${ep}_seed${seed}_e10_top30_s300_success_rerank"
  local policy="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" wm_experiment_scripts/success_aligned_rerank_pilot.py \
    --policy "$policy" \
    --output "$OUTDIR/${tag}.json" \
    --num-eval 10 \
    --topk 30 \
    --num-samples 300 \
    --cem-steps 30 \
    --seed "$seed" > "$OUTDIR/${tag}.log" 2>&1
  echo "[DONE] ${tag}"
}

(
  run_one 0 4 42
  run_one 0 4 43
) > "$OUTDIR/gpu0_ep4.log" 2>&1 &

(
  run_one 1 9 42
  run_one 1 9 43
) > "$OUTDIR/gpu1_ep9.log" 2>&1 &

(
  run_one 2 16 42
  run_one 2 16 43
) > "$OUTDIR/gpu2_ep16.log" 2>&1 &

wait

"$PY" - <<'PY'
import json
from pathlib import Path

outdir = Path("/data1/jingyixi/wm_runs/success_aligned_rerank")
rows = []
for path in sorted(outdir.glob("gate07_ep*_seed*_e10_top30_s300_success_rerank.json")):
    data = json.load(open(path))
    rows.append({
        "file": path.name,
        "top1": data.get("top1_success_rate"),
        "oracle": data.get("oracle_topk_success_rate"),
        "rerank": data.get("success_aligned_rerank", {}).get("success_rate"),
        "cost_gap_failure_minus_success": data.get("cost_success_diagnostic", {}).get("failure_minus_success"),
    })
lines = [
    "|file|top1|oracle|success_rerank|failure_minus_success_cost|",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    def fmt(x):
        return "NA" if x is None else f"{x:.1f}"
    lines.append(
        f"|{r['file']}|{fmt(r['top1'])}|{fmt(r['oracle'])}|{fmt(r['rerank'])}|{fmt(r['cost_gap_failure_minus_success'])}|"
    )
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
