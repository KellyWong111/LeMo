#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/cem_success_cost_eval
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
  local gpu=$1
  local tag=$2
  local ep=$3
  local policy_dir="pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_cemsuccess_${tag}"
  local model="lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_cemsuccess_${tag}_epoch_${ep}"
  local ckpt="$STABLE/$policy_dir/${model}_object.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] missing $ckpt"
    return 0
  fi
  local out="eval_${tag}_ep${ep}_seed42_h4_s1000_k100_n20.txt"
  echo "[EVAL] gpu=${gpu} tag=${tag} ep=${ep}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="${policy_dir}/${model}" \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed=42 \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_${tag}_ep${ep}.log" 2>&1
  echo "[DONE] ${tag} ep=${ep}"
}

(
  for ep in 50 100 150 200; do run_eval 2 pred_lr1e5_a1e4 "$ep"; done
) > "$OUTDIR/gpu2_pred.log" 2>&1 &

(
  for ep in 50 100 150 200; do run_eval 3 predact_lr5e6_a1e4 "$ep"; done
) > "$OUTDIR/gpu3_predact.log" 2>&1 &

wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path("/data1/jingyixi/.stable_worldmodel")
outdir = Path("/data1/jingyixi/wm_runs/cem_success_cost_eval")
rows = []
for d in stable.glob("pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_cemsuccess_*"):
    tag = d.name.split("gate07_cemsuccess_", 1)[-1]
    for p in d.glob("eval_*_seed42_h4_s1000_k100_n20.txt"):
        text = p.read_text(errors="ignore")
        m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text)
        if not m:
            m = re.search(r"'success_rate':\s*([0-9.]+)", text)
        epm = re.search(r"_ep(\d+)_seed42_", p.name)
        rows.append({
            "tag": tag,
            "epoch": int(epm.group(1)) if epm else None,
            "success_rate": float(m.group(1)) if m else None,
            "file": str(p),
        })
rows.sort(key=lambda r: (r["tag"], r["epoch"] or -1))
(outdir / "summary.json").write_text(json.dumps(rows, indent=2))
lines = ["|tag|epoch|success_rate|", "|---|---:|---:|"]
for r in rows:
    val = "NA" if r["success_rate"] is None else f"{r['success_rate']:.1f}"
    lines.append(f"|{r['tag']}|{r['epoch']}|{val}|")
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
