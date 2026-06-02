from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base
from topk_oracle_pilot import eval_fixed_plans, get_multistart_topk_candidates


POLICIES = {
    "official_ep13": "pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13",
    "gate07_ep4": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4",
    "stateroll_l003_ep1": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1",
}


def clone_prepared(prepared):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared.items()}


def pairwise_mean_flat(x: torch.Tensor, max_items: int = 30) -> float:
    x = x.detach().float().cpu().reshape(x.shape[0], x.shape[1], -1)
    vals = []
    for row in x:
        row = row[:max_items]
        if row.shape[0] < 2:
            continue
        d = torch.cdist(row, row)
        iu = torch.triu_indices(row.shape[0], row.shape[0], offset=1)
        vals.append(float(d[iu[0], iu[1]].mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


@torch.inference_mode()
def rollout_metrics(model, prepared_base, candidates: torch.Tensor):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    costs = model.get_cost(prepared, candidates.to("cuda").float()).detach().float().cpu().numpy()
    pred = prepared.get("predicted_emb")
    out = {"costs": costs}
    if pred is not None:
        pred = pred.detach().float().cpu()
        final = pred[:, :, -1, :]
        out["rollout_final_spread_mean"] = pairwise_mean_flat(final)
        step = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).norm(dim=-1)
        out["rollout_step_norm_mean"] = float(step.mean().item())
        out["rollout_step_norm_std"] = float(step.std().item())
    return out


def success_cost_metrics(labels: np.ndarray, costs: np.ndarray):
    gaps, aucs = [], []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) and len(neg):
            gaps.append(float(neg.mean() - pos.mean()))
            aucs.append(float((pos[:, None] < neg[None, :]).mean()))
    return {
        "cost_gap": None if not gaps else float(np.mean(gaps)),
        "pairwise_success_auc": None if not aucs else float(np.mean(aucs)),
    }


def evaluate_policy(name: str, policy: str, seed: int, args):
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = policy
    cfg.eval.num_eval = args.num_eval
    cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    OmegaConf.update(cfg, "plan_config.horizon", args.horizon, merge=True)
    OmegaConf.update(cfg, "plan_config.action_block", args.action_block, merge=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", args.receding_horizon, merge=True)

    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]

    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared_base = base.make_eval_like_info(
        raw_info,
        {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)},
        process,
    )

    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

    model = base.load_model(cfg, cache_dir=args.cache_dir)
    candidates, topk_costs = get_multistart_topk_candidates(
        model,
        prepared_base,
        action_dim=action_dim,
        horizon=int(cfg.plan_config.horizon),
        num_samples=args.num_samples,
        topk=args.topk,
        n_steps=args.cem_steps,
        seed=seed,
        restarts=args.restarts,
    )
    roll = rollout_metrics(model, prepared_base, candidates.float())
    del model
    torch.cuda.empty_cache()

    plans = candidates.numpy()
    labels = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(labels, axis=1)
    top1 = float(labels[:, 0].mean() * 100.0)
    oracle = float(labels.any(axis=1).mean() * 100.0)
    first_success_rank = []
    for row in labels:
        hits = np.nonzero(row)[0]
        first_success_rank.append(int(hits[0]) if len(hits) else None)
    costs = roll["costs"]
    top2_margin = topk_costs[:, 1] - topk_costs[:, 0] if topk_costs.shape[1] > 1 else torch.zeros(topk_costs.shape[0])
    top5_margin = topk_costs[:, min(4, topk_costs.shape[1] - 1)] - topk_costs[:, 0]
    return {
        "name": name,
        "policy": policy,
        "seed": seed,
        "settings": {
            "num_eval": args.num_eval,
            "topk": args.topk,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "restarts": args.restarts,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "receding_horizon": args.receding_horizon,
        },
        "top1_success": top1,
        "oracle_topk_success": oracle,
        "oracle_gap": oracle - top1,
        "action_spread_mean": pairwise_mean_flat(candidates),
        "top2_margin_mean": float(top2_margin.mean().item()),
        "top5_margin_mean": float(top5_margin.mean().item()),
        "mean_first_success_rank": None if not [x for x in first_success_rank if x is not None] else float(np.mean([x for x in first_success_rank if x is not None])),
        "first_success_rank": first_success_rank,
        **{k: v for k, v in roll.items() if k != "costs"},
        **success_cost_metrics(labels, costs),
    }


def fmt(x, digits=1):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def summarize(rows):
    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    lines = [
        "# Candidate pool coverage comparison",
        "",
        "|policy|n|top1|oracle_topk|oracle_gap|AUC|cost_gap|action_spread|rollout_spread|top2_margin|top5_margin|mean_first_success_rank|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ["top1_success", "oracle_topk_success", "oracle_gap", "pairwise_success_auc", "cost_gap", "action_spread_mean", "rollout_final_spread_mean", "top2_margin_mean", "top5_margin_mean", "mean_first_success_rank"]
    for name, items in sorted(by_name.items()):
        vals = {}
        for key in keys:
            xs = [x.get(key) for x in items if x.get(key) is not None]
            vals[key] = None if not xs else float(np.mean(xs))
        lines.append(
            f"|{name}|{len(items)}|{fmt(vals['top1_success'])}|{fmt(vals['oracle_topk_success'])}|{fmt(vals['oracle_gap'])}|"
            f"{fmt(vals['pairwise_success_auc'],3)}|{fmt(vals['cost_gap'],3)}|{fmt(vals['action_spread_mean'],3)}|"
            f"{fmt(vals['rollout_final_spread_mean'],3)}|{fmt(vals['top2_margin_mean'],3)}|{fmt(vals['top5_margin_mean'],3)}|{fmt(vals['mean_first_success_rank'],2)}|"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME", "/data1/jingyixi/.stable_worldmodel"))
    parser.add_argument("--policies", default="official_ep13,gate07_ep4,stateroll_l003_ep1")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=4)
    args = parser.parse_args()

    names = [x for x in args.policies.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw_results.json"
    rows = []
    if raw_path.exists():
        rows = json.loads(raw_path.read_text())
    done = {(r["name"], int(r["seed"])) for r in rows}
    for seed in seeds:
        for name in names:
            if (name, seed) in done:
                print(f"[SKIP] {name} seed={seed}", flush=True)
                continue
            print(f"[RUN] {name} seed={seed}", flush=True)
            rows.append(evaluate_policy(name, POLICIES[name], seed, args))
            raw_path.write_text(json.dumps(rows, indent=2))
            (outdir / "summary.md").write_text(summarize(rows))
    print(summarize(rows))


if __name__ == "__main__":
    main()
