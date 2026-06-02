#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUTDIR=/data1/jingyixi/wm_runs/success_critic_cem
mkdir -p "$OUTDIR"

export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
cd "$ROOT"

CRITIC="$OUTDIR/ep9_s42_43_44_n30_top30_scalar_critic.pt"

echo "[TRAIN CRITIC]"
CUDA_VISIBLE_DEVICES=2 "$PY" wm_experiment_scripts/train_cem_success_critic.py \
  --policy pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_9 \
  --output "$CRITIC" \
  --train-seeds 42,43,44 \
  --num-eval 30 \
  --topk 30 \
  --num-samples 300 \
  --cem-steps 30 \
  --epochs 300 \
  --lr 1e-3 \
  > "$OUTDIR/train_critic.log" 2>&1

echo "[EVAL CRITIC CEM]"
run_eval() {
  local gpu=$1
  local w=$2
  local out="criticcem_w${w}_ep9_seed42_h4_s1000_k100_n20.txt"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_9 \
    cache_dir="$STABLE" \
    eval.num_eval=20 \
    seed=42 \
    solver._target_=cem_success_critic_solver.SuccessCriticCEMSolver \
    solver.critic_path="$CRITIC" \
    solver.critic_weight="$w" \
    solver.num_samples=1000 \
    solver.topk=100 \
    solver.n_steps=20 \
    plan_config.horizon=4 \
    plan_config.action_block=5 \
    plan_config.receding_horizon=4 \
    output.filename="$out" > "$OUTDIR/eval_w${w}.log" 2>&1
}

run_eval 2 0.25 &
run_eval 3 0.50 &
wait
run_eval 2 1.00 &
run_eval 3 2.00 &
wait

"$PY" - <<'PY'
import json
import re
from pathlib import Path

stable = Path("/data1/jingyixi/.stable_worldmodel/pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07")
outdir = Path("/data1/jingyixi/wm_runs/success_critic_cem")
rows = []
for p in stable.glob("criticcem_w*_ep9_seed42_h4_s1000_k100_n20.txt"):
    text = p.read_text(errors="ignore")
    m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text)
    if not m:
        m = re.search(r"'success_rate':\s*([0-9.]+)", text)
    wm = re.search(r"criticcem_w([^_]+)_", p.name)
    rows.append({
        "weight": wm.group(1) if wm else None,
        "success_rate": float(m.group(1)) if m else None,
        "file": str(p),
    })
rows.sort(key=lambda r: float(r["weight"]))
(outdir / "summary.json").write_text(json.dumps(rows, indent=2))
lines = ["|critic_weight|success_rate|", "|---:|---:|"]
for r in rows:
    val = "NA" if r["success_rate"] is None else f"{r['success_rate']:.1f}"
    lines.append(f"|{r['weight']}|{val}|")
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
