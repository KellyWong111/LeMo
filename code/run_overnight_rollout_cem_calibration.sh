#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/overnight_rollout_cem_calibration
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"
POLICY_ROOT=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07
RESULT_ROOT="$STABLE/$POLICY_ROOT"

run_one() {
  local gpu=$1
  local ep=$2
  local seed=$3
  local horizon=$4
  local samples=$5
  local topk=$6
  local steps=$7
  local tag="gate07_ep${ep}_seed${seed}_h${horizon}_rh${horizon}_b5_s${samples}_k${topk}_n${steps}"
  local policy="${POLICY_ROOT}/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  local log="${OUTDIR}/${tag}.log"
  local out="${tag}.txt"

  if [[ -f "${RESULT_ROOT}/${out}" ]]; then
    echo "[SKIP] ${tag}"
    return
  fi

  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed="$seed" \
    solver.num_samples="$samples" \
    solver.topk="$topk" \
    solver.n_steps="$steps" \
    plan_config.horizon="$horizon" \
    plan_config.action_block=5 \
    plan_config.receding_horizon="$horizon" \
    output.filename="$out" > "$log" 2>&1
  echo "[DONE] ${tag}"
}

summarize() {
  "$PY" - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

stable = Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07")
outdir = Path("/data1/jingyixi/wm_runs/overnight_rollout_cem_calibration")

def parse_success(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    if not m:
        m = re.search(r'"success_rate":\s*([0-9.]+)', txt)
    return float(m.group(1)) if m else None

rows = []
for path in stable.glob("gate07_ep*_seed*_h*_rh*_b5_s*_k*_n*.txt"):
    name = path.stem
    m = re.search(r"gate07_ep(\d+)_seed(\d+)_h(\d+)_rh\d+_b5_s(\d+)_k(\d+)_n(\d+)", name)
    if not m:
        continue
    ep, seed, h, samples, topk, steps = map(int, m.groups())
    if ep not in [4, 9, 16] or h not in [3, 4]:
        continue
    if (samples, topk, steps) not in [(600, 60, 20), (1000, 100, 20), (1000, 100, 30)]:
        continue
    rows.append({
        "epoch": ep,
        "seed": seed,
        "horizon": h,
        "rollout_steps": h * 5,
        "samples": samples,
        "topk": topk,
        "n_steps": steps,
        "success_rate": parse_success(path),
        "file": str(path),
    })

rows = sorted(rows, key=lambda r: (r["epoch"], r["horizon"], r["samples"], r["topk"], r["n_steps"], r["seed"]))
summary = defaultdict(list)
for r in rows:
    key = f"ep{r['epoch']}_h{r['horizon']}_s{r['samples']}_k{r['topk']}_n{r['n_steps']}"
    if r["success_rate"] is not None:
        summary[key].append(r["success_rate"])

summary_rows = []
for key, vals in sorted(summary.items()):
    summary_rows.append({
        "setting": key,
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    })

outdir.mkdir(parents=True, exist_ok=True)
(outdir / "raw_results.json").write_text(json.dumps(rows, indent=2))
(outdir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
lines = ["|setting|mean|min|max|n|", "|---|---:|---:|---:|---:|"]
for r in summary_rows:
    lines.append(f"|{r['setting']}|{r['mean']:.1f}|{r['min']:.1f}|{r['max']:.1f}|{r['n']}|")
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

(
  run_one 0 4 42 3 600 60 20
  run_one 0 4 43 3 600 60 20
  run_one 0 4 44 3 600 60 20
  run_one 0 4 42 3 1000 100 20
  run_one 0 4 43 3 1000 100 20
  run_one 0 4 44 3 1000 100 20
  run_one 0 4 42 3 1000 100 30
  run_one 0 4 43 3 1000 100 30
  run_one 0 4 44 3 1000 100 30
) > "$OUTDIR/gpu0_queue.log" 2>&1 &

(
  run_one 1 4 42 4 600 60 20
  run_one 1 4 43 4 600 60 20
  run_one 1 4 44 4 600 60 20
  run_one 1 4 42 4 1000 100 20
  run_one 1 4 43 4 1000 100 20
  run_one 1 4 44 4 1000 100 20
  run_one 1 4 42 4 1000 100 30
  run_one 1 4 43 4 1000 100 30
  run_one 1 4 44 4 1000 100 30
) > "$OUTDIR/gpu1_queue.log" 2>&1 &

(
  run_one 2 9 42 3 600 60 20
  run_one 2 9 43 3 600 60 20
  run_one 2 9 44 3 600 60 20
  run_one 2 9 42 3 1000 100 20
  run_one 2 9 43 3 1000 100 20
  run_one 2 9 44 3 1000 100 20
  run_one 2 16 42 3 600 60 20
  run_one 2 16 43 3 600 60 20
  run_one 2 16 44 3 600 60 20
  run_one 2 16 42 3 1000 100 20
  run_one 2 16 43 3 1000 100 20
  run_one 2 16 44 3 1000 100 20
) > "$OUTDIR/gpu2_queue.log" 2>&1 &

(
  run_one 3 9 42 4 600 60 20
  run_one 3 9 43 4 600 60 20
  run_one 3 9 44 4 600 60 20
  run_one 3 9 42 4 1000 100 20
  run_one 3 9 43 4 1000 100 20
  run_one 3 9 44 4 1000 100 20
  run_one 3 16 42 4 600 60 20
  run_one 3 16 43 4 600 60 20
  run_one 3 16 44 4 600 60 20
  run_one 3 16 42 4 1000 100 20
  run_one 3 16 43 4 1000 100 20
  run_one 3 16 44 4 1000 100 20
) > "$OUTDIR/gpu3_queue.log" 2>&1 &

wait
echo "[ALL DONE]"
summarize
