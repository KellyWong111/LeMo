#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/reachability_contrast_eval
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl

cd "$ROOT"

echo "[WAIT] waiting for reachability contrast training to finish"
while pgrep -f "train_encoder_moda_rank_full.py.*gate07_reach_w" >/dev/null; do
  sleep 120
done
echo "[START] training finished; launching planning eval"

run_eval() {
  local gpu=$1
  local tag=$2
  local ep=$3
  local policy_dir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_reach_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_reach_${tag}_epoch_${ep}"
  local policy="${policy_dir}/${model}"
  local result_dir="$STABLE/$policy_dir"
  local ckpt="$result_dir/${model}_object.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] missing $ckpt"
    return 0
  fi
  local out="eval_${tag}_ep${ep}_seed42_h4_s1000_k100_n20.txt"
  local log="$OUTDIR/eval_${tag}_ep${ep}.log"
  echo "[EVAL] gpu=${gpu} ${tag} epoch=${ep}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$policy" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed=42 \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$log" 2>&1
}

eval_queue() {
  local gpu=$1
  local tag=$2
  for ep in 1 2 4 8 12; do
    run_eval "$gpu" "$tag" "$ep"
  done
}

eval_queue 0 w003_t010_cos > "$OUTDIR/gpu0_w003_t010_cos.log" 2>&1 &
eval_queue 1 w010_t010_cos > "$OUTDIR/gpu1_w010_t010_cos.log" 2>&1 &
eval_queue 2 w003_t005_cos > "$OUTDIR/gpu2_w003_t005_cos.log" 2>&1 &
eval_queue 3 w010_t005_cos > "$OUTDIR/gpu3_w010_t005_cos.log" 2>&1 &

wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path
from collections import defaultdict

stable = Path("/data1/jingyixi/.stable_worldmodel")
outdir = Path("/data1/jingyixi/wm_runs/reachability_contrast_eval")
rows = []
for result_dir in stable.glob("pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_reach_*"):
    tag = result_dir.name.split("gate07_reach_", 1)[-1]
    for path in result_dir.glob("eval_*_seed42_h4_s1000_k100_n20.txt"):
        text = path.read_text(errors="ignore")
        m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text)
        if not m:
            m = re.search(r"'success_rate':\s*([0-9.]+)", text)
        epm = re.search(r"_ep(\d+)_seed42_", path.name)
        rows.append({
            "tag": tag,
            "epoch": int(epm.group(1)) if epm else None,
            "success_rate": float(m.group(1)) if m else None,
            "file": str(path),
        })
rows.sort(key=lambda r: (r["tag"], r["epoch"] if r["epoch"] is not None else -1))
(outdir / "summary.json").write_text(json.dumps(rows, indent=2))
lines = ["|tag|epoch|success_rate|", "|---|---:|---:|"]
for row in rows:
    val = "NA" if row["success_rate"] is None else f"{row['success_rate']:.1f}"
    lines.append(f"|{row['tag']}|{row['epoch']}|{val}|")
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
