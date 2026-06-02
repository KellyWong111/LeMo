#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/condensed_rollout_cem_pilot
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
  local gpu=$1 ep=$2 seed=$3 horizon=$4 samples=$5 topk=$6 steps=$7
  local tag="pilot_e10_gate07_ep${ep}_seed${seed}_h${horizon}_rh${horizon}_b5_s${samples}_k${topk}_n${steps}"
  local out="${tag}.txt"
  local log="$OUTDIR/${tag}.log"
  local policy="${POLICY_ROOT}/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_${ep}"
  if [[ -f "${RESULT_ROOT}/${out}" ]]; then
    echo "[SKIP] ${tag}"
    return
  fi
  echo "[RUN] gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" cache_dir="$STABLE" eval.num_eval=10 seed="$seed" \
    solver.num_samples="$samples" solver.topk="$topk" solver.n_steps="$steps" \
    plan_config.horizon="$horizon" plan_config.action_block=5 \
    plan_config.receding_horizon="$horizon" output.filename="$out" > "$log" 2>&1
  echo "[DONE] ${tag}"
}

summarize() {
  "$PY" - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

stable = Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07")
outdir = Path("/data1/jingyixi/wm_runs/condensed_rollout_cem_pilot")

def parse(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None

rows = []
for path in stable.glob("pilot_e10_gate07_ep*_seed*_h*_rh*_b5_s*_k*_n*.txt"):
    m = re.search(r"pilot_e10_gate07_ep(\d+)_seed(\d+)_h(\d+)_rh\d+_b5_s(\d+)_k(\d+)_n(\d+)", path.stem)
    if not m:
        continue
    ep, seed, h, samples, topk, steps = map(int, m.groups())
    rows.append({
        "epoch": ep,
        "seed": seed,
        "horizon": h,
        "rollout_steps": h * 5,
        "samples": samples,
        "topk": topk,
        "n_steps": steps,
        "success_rate": parse(path),
        "file": str(path),
    })
rows = sorted(rows, key=lambda r: (r["samples"], r["n_steps"], r["epoch"], r["horizon"], r["seed"]))
groups = defaultdict(list)
for r in rows:
    key = f"h{r['horizon']}_s{r['samples']}_k{r['topk']}_n{r['n_steps']}"
    if r["success_rate"] is not None:
        groups[key].append(r["success_rate"])
summary = []
for key, vals in sorted(groups.items()):
    summary.append({"setting": key, "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)})
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "raw_results.json").write_text(json.dumps(rows, indent=2))
(outdir / "summary.json").write_text(json.dumps(summary, indent=2))
lines = ["|setting|mean|min|max|n|", "|---|---:|---:|---:|---:|"]
for r in summary:
    lines.append(f"|{r['setting']}|{r['mean']:.1f}|{r['min']:.1f}|{r['max']:.1f}|{r['n']}|")
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

# Queue design:
# - num_eval=10 pilot so it finishes overnight.
# - ep4/ep9/ep16, seeds 42/43, horizons h3/h4.
# - compare moderate search and stronger search without n30 first.
(
  for ep in 4 9 16; do
    for seed in 42 43; do
      run_one 0 "$ep" "$seed" 3 600 60 20
    done
  done
) > "$OUTDIR/gpu0_queue.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43; do
      run_one 1 "$ep" "$seed" 4 600 60 20
    done
  done
) > "$OUTDIR/gpu1_queue.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43; do
      run_one 2 "$ep" "$seed" 3 1000 100 20
    done
  done
) > "$OUTDIR/gpu2_queue.log" 2>&1 &

(
  for ep in 4 9 16; do
    for seed in 42 43; do
      run_one 3 "$ep" "$seed" 4 1000 100 20
    done
  done
) > "$OUTDIR/gpu3_queue.log" 2>&1 &

wait
echo "[ALL DONE]"
summarize
