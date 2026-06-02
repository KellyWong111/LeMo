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


@torch.inference_mode()
def consistency_scores(
    model,
    prepared_base,
    candidates: torch.Tensor,
    base_costs: torch.Tensor,
    sigma: float,
    repeats: int,
    seed: int,
):
    device = "cuda"
    num_envs, topk = candidates.shape[:2]
    gen = torch.Generator(device=device).manual_seed(seed)
    costs = []
    for i in range(repeats):
        noisy = candidates.to(device)
        if sigma > 0:
            noisy = noisy + sigma * torch.randn(
                noisy.shape,
                generator=gen,
                device=device,
                dtype=noisy.dtype,
            )
        prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
        prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
        costs.append(model.get_cost(prepared, noisy).detach().cpu())
    cost_stack = torch.stack(costs, dim=0)
    return {
        "base": base_costs.float(),
        "mean": cost_stack.mean(dim=0).float(),
        "std": cost_stack.std(dim=0).float(),
    }


def first_success_rank(oracle_success: np.ndarray):
    ranks = []
    for row in oracle_success:
        hits = np.nonzero(row)[0]
        ranks.append(int(hits[0]) if len(hits) else None)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)

    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]

    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    transform = {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)}
    prepared_base = base.make_eval_like_info(raw_info, transform, process)

    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

    model = base.load_model(cfg, cache_dir=None)
    candidates, topk_costs = get_multistart_topk_candidates(
        model,
        prepared_base,
        action_dim=action_dim,
        horizon=int(cfg.plan_config.horizon),
        num_samples=args.num_samples,
        topk=args.topk,
        n_steps=args.cem_steps,
        seed=args.seed,
        restarts=args.restarts,
    )
    score_parts = consistency_scores(
        model,
        prepared_base,
        candidates,
        topk_costs,
        sigma=args.sigma,
        repeats=args.repeats,
        seed=args.seed + 777,
    )
    del model
    torch.cuda.empty_cache()

    plans = candidates.numpy()
    top1_metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, 0])

    oracle_success = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        oracle_success.append(np.asarray(metrics["episode_successes"], dtype=bool))
    oracle_success = np.stack(oracle_success, axis=1)

    result = {
        "policy": args.policy,
        "indices": indices.tolist(),
        "settings": vars(args),
        "baseline_top1_success_rate": float(np.mean(top1_metrics["episode_successes"]) * 100.0),
        "oracle_topk_success_rate": float(np.mean(oracle_success.any(axis=1)) * 100.0),
        "baseline_top1_episode_successes": np.asarray(top1_metrics["episode_successes"], dtype=bool).tolist(),
        "oracle_episode_successes": oracle_success.any(axis=1).tolist(),
        "oracle_first_success_rank": first_success_rank(oracle_success),
        "rerank": {},
        "score_stats": {
            "base_mean": float(score_parts["base"].mean()),
            "perturb_mean_mean": float(score_parts["mean"].mean()),
            "perturb_std_mean": float(score_parts["std"].mean()),
            "perturb_std_top1_mean": float(score_parts["std"][:, 0].mean()),
        },
    }

    batch = torch.arange(args.num_eval)
    for lam in args.lambdas:
        score = score_parts["mean"] + lam * score_parts["std"]
        chosen = torch.argmin(score, dim=1)
        chosen_plans = candidates[batch, chosen].numpy()
        metrics = eval_fixed_plans(cfg, dataset, process, indices, chosen_plans)
        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        result["rerank"][str(lam)] = {
            "success_rate": float(np.mean(successes) * 100.0),
            "episode_successes": successes.tolist(),
            "chosen_rank": chosen.tolist(),
            "chosen_base_cost": score_parts["base"][batch, chosen].tolist(),
            "chosen_cost_std": score_parts["std"][batch, chosen].tolist(),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
