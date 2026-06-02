#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean
PY=/data1/jingyixi/conda_envs/lewm5090/bin/python
STABLE=/data1/jingyixi/.stable_worldmodel
OUT=/data1/jingyixi/wm_runs/state_rollout_confirm_main
mkdir -p "$OUT"
export STABLEWM_HOME=$STABLE
export PYTHONPATH=$ROOT:$ROOT/wm_experiment_scripts
export MODA_TRITON_ROOT=/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
export TMPDIR=/data1/jingyixi/tmp/tmp
export XDG_CACHE_HOME=/data1/jingyixi/tmp/xdg
export TRITON_CACHE_DIR=/data1/jingyixi/tmp/triton
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
BASE_PREFIX=pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07
BASE_CKPT="$STABLE/$BASE_PREFIX/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4_object.ckpt"
L003_SUB=${BASE_PREFIX}_staterollseq_l003
L003_MODEL=lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003

policy_for(){
  local kind=$1 ep=${2:-}
  if [[ "$kind" == baseline ]]; then echo "$BASE_PREFIX/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4"; else echo "$L003_SUB/${L003_MODEL}_epoch_${ep}"; fi
}
outdir_for(){ if [[ "$1" == baseline ]]; then echo "$STABLE/$BASE_PREFIX"; else echo "$STABLE/$L003_SUB"; fi; }
outname_for(){
  local kind=$1 ep=$2 seed=$3
  if [[ "$kind" == baseline ]]; then echo "gate07_epoch4_standard_seed${seed}_h4_s300_k30_n30.txt"; else echo "staterollseq_l003_ep${ep}_standard_seed${seed}_h4_s300_k30_n30.txt"; fi
}

eval_one(){
  local gpu=$1 kind=$2 ep=$3 seed=$4
  local outname dir pol
  outname=$(outname_for "$kind" "$ep" "$seed")
  dir=$(outdir_for "$kind")
  pol=$(policy_for "$kind" "$ep")
  if [[ -f "$dir/$outname" ]]; then echo "[SKIP_EVAL] $kind ep$ep seed$seed"; return 0; fi
  echo "[EVAL] $(date -Is) gpu=$gpu kind=$kind ep=$ep seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval.py --config-name=pusht.yaml \
    policy="$pol" cache_dir="$STABLE" eval.num_eval=20 seed="$seed" \
    solver.num_samples=300 solver.topk=30 solver.n_steps=30 \
    plan_config.horizon=4 plan_config.action_block=5 plan_config.receding_horizon=4 \
    output.filename="$outname" > "$OUT/eval_${kind}_ep${ep}_seed${seed}.log" 2>&1
  echo "[EVAL_DONE] $(date -Is) gpu=$gpu kind=$kind ep=$ep seed=$seed"
}

train_l003_ep4(){
  if [[ -f "$STABLE/$L003_SUB/${L003_MODEL}_epoch_4_object.ckpt" ]]; then echo "[SKIP_TRAIN_EP4] exists"; return 0; fi
  echo "[TRAIN_EP4] $(date -Is) gpu=3"
  CUDA_VISIBLE_DEVICES=3 "$PY" train_encoder_moda_rank_full.py \
    --config-name=lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07 \
    subdir="$L003_SUB" output_model_name="$L003_MODEL" \
    warm_start_ckpt="$BASE_CKPT" warm_start_strict=false \
    trainer.devices=1 trainer.accelerator=gpu trainer.max_epochs=4 \
    +trainer.limit_train_batches=300 +trainer.limit_val_batches=30 \
    trainer.precision=bf16 wandb.enabled=false loader.batch_size=32 loader.num_workers=4 \
    +train.freeze_encoder=true +train.freeze_projector=true \
    +loss.state_rollout.enabled=true +loss.state_rollout.weight=0.03 +loss.state_rollout.hidden_dim=256 \
    > "$OUT/train_l003_to_ep4.log" 2>&1
  echo "[TRAIN_EP4_DONE] $(date -Is)"
}

collect(){
  "$PY" /data1/jingyixi/wm_runs/state_rollout_confirm_collect.py
}

worker_a(){
  for seed in 42 43 44 45 46 47; do eval_one 2 baseline 0 "$seed"; done
  for seed in 45 46 47; do eval_one 2 l003 1 "$seed"; done
}
worker_b(){
  train_l003_ep4
  for seed in 42 43 44; do eval_one 3 l003 4 "$seed"; done
}
worker_c(){
  # Cost-success diagnostic: same eval seeds 42/43/44, top30 CEM candidates per policy.
  for seed in 42 43 44; do
    for kind in baseline l003; do
      local pol tag
      if [[ "$kind" == baseline ]]; then pol=$(policy_for baseline 0); tag=baseline_gate07_ep4; else pol=$(policy_for l003 1); tag=stateroll_l003_ep1; fi
      local out="$OUT/cost_success_${tag}_seed${seed}.json"
      if [[ -f "$out" ]]; then echo "[SKIP_DIAG] $tag seed$seed"; continue; fi
      echo "[DIAG] $(date -Is) gpu=3 tag=$tag seed=$seed"
      CUDA_VISIBLE_DEVICES=3 "$PY" wm_experiment_scripts/topk_oracle_pilot.py \
        --policy "$pol" --output "$out" --num-eval 20 --topk 30 --num-samples 300 --cem-steps 30 --seed "$seed" --restarts 1 \
        > "$OUT/diag_${tag}_seed${seed}.log" 2>&1
      echo "[DIAG_DONE] $(date -Is) gpu=3 tag=$tag seed=$seed"
    done
  done
}
case "${1:-all}" in
  a) worker_a ;;
  b) worker_b ;;
  c) worker_c ;;
  collect) collect ;;
  all)
    echo "[START] $(date -Is)" | tee "$OUT/master.log"
    (worker_a > "$OUT/worker_a.log" 2>&1; echo a_done >> "$OUT/master.log") & echo $! > "$OUT/worker_a.pid"
    (worker_b > "$OUT/worker_b.log" 2>&1; echo b_done >> "$OUT/master.log") & echo $! > "$OUT/worker_b.pid"
    wait
    # Run diagnostic after B to avoid fighting training on GPU3.
    worker_c > "$OUT/worker_c.log" 2>&1 || true
    collect > "$OUT/collect.log" 2>&1 || true
    cat "$OUT/summary.md" 2>/dev/null || true
    ;;
  *) echo usage: $0 '[all|a|b|c|collect]' >&2; exit 2 ;;
esac
