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


def pairwise_spread(x: torch.Tensor) -> torch.Tensor:
    vals = []
    x = x.detach().float().cpu()
    for env in x:
        d = torch.cdist(env, env)
        iu = torch.triu_indices(env.shape[0], env.shape[0], offset=1)
        vals.append(d[iu[0], iu[1]].mean())
    return torch.stack(vals)


def topk_emb_dist(final_emb: torch.Tensor, costs: torch.Tensor, k: int = 5) -> dict[str, list[float]]:
    order = torch.argsort(costs.detach().cpu(), dim=1)
    final = final_emb.detach().float().cpu()
    top1 = final[torch.arange(final.shape[0]), order[:, 0]]
    top2 = final[torch.arange(final.shape[0]), order[:, 1]]
    kth = final[torch.arange(final.shape[0]), order[:, min(k - 1, order.shape[1] - 1)]]
    return {
        "top1_top2_emb_l2": (top2 - top1).norm(dim=-1).tolist(),
        f"top1_top{k}_emb_l2": (kth - top1).norm(dim=-1).tolist(),
    }


def summarize_step(costs: torch.Tensor, candidates: torch.Tensor, topk: int) -> dict[str, list[float]]:
    costs_cpu = costs.detach().float().cpu()
    sorted_costs, topk_idx = torch.topk(costs, k=topk, dim=1, largest=False)
    sorted_cpu = torch.sort(costs_cpu, dim=1).values
    batch_idx = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
    elite = candidates[batch_idx, topk_idx]
    return {
        "top1_cost": sorted_cpu[:, 0].tolist(),
        "top2_margin": (sorted_cpu[:, 1] - sorted_cpu[:, 0]).tolist(),
        "top5_margin": (sorted_cpu[:, min(4, sorted_cpu.shape[1] - 1)] - sorted_cpu[:, 0]).tolist(),
        "cost_std": costs_cpu.std(dim=1).tolist(),
        "elite_cost_mean": sorted_costs.detach().float().cpu().mean(dim=1).tolist(),
        "elite_action_std": elite.detach().float().cpu().reshape(elite.shape[0], elite.shape[1], -1).std(dim=1).mean(dim=1).tolist(),
    }


@torch.inference_mode()
def trace_cem(model, prepared_base, action_dim: int, horizon: int, num_samples: int, topk: int, n_steps: int, seed: int):
    device = "cuda"
    num_envs = next(v for v in prepared_base.values() if torch.is_tensor(v)).shape[0]
    mean = torch.zeros(num_envs, horizon, action_dim, device=device)
    var = torch.ones(num_envs, horizon, action_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    step_records = []
    final_costs = None
    final_candidates = None
    final_prepared = None

    for step in range(n_steps):
        candidates = torch.randn(
            num_envs,
            num_samples,
            horizon,
            action_dim,
            generator=gen,
            device=device,
        )
        candidates = candidates * var[:, None] + mean[:, None]
        candidates[:, 0] = mean

        prepared = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in prepared_base.items()
        }
        prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
        costs = model.get_cost(prepared, candidates)
        step_record = summarize_step(costs, candidates, topk=topk)
        step_record["step"] = step
        step_record["mean_action_norm"] = mean.detach().float().cpu().reshape(num_envs, -1).norm(dim=1).tolist()
        step_record["var_mean"] = var.detach().float().cpu().reshape(num_envs, -1).mean(dim=1).tolist()
        step_records.append(step_record)

        topk_vals, topk_idx = torch.topk(costs, k=topk, dim=1, largest=False)
        batch_idx = torch.arange(num_envs, device=device)[:, None]
        elite = candidates[batch_idx, topk_idx]
        new_mean = elite.mean(dim=1)
        new_var = elite.std(dim=1)
        step_record["mean_update_norm"] = (new_mean - mean).detach().float().cpu().reshape(num_envs, -1).norm(dim=1).tolist()
        mean, var = new_mean, new_var

        final_costs = costs.detach().cpu()
        final_candidates = candidates.detach()
        final_prepared = prepared

    pred = final_prepared["predicted_emb"].detach().float().cpu()
    final_emb = pred[:, :, -1]
    trajectory_emb = pred.reshape(pred.shape[0], pred.shape[1], -1)
    final_sorted, final_order = torch.sort(final_costs, dim=1)
    topk_order = final_order[:, : min(5, final_order.shape[1])]
    batch_cpu = torch.arange(final_emb.shape[0])[:, None]
    topk_final = final_emb[batch_cpu, topk_order]
    topk_traj = trajectory_emb[batch_cpu, topk_order]

    final_summary = {
        "top1_cost": final_sorted[:, 0].tolist(),
        "top2_margin": (final_sorted[:, 1] - final_sorted[:, 0]).tolist(),
        "top5_margin": (final_sorted[:, min(4, final_sorted.shape[1] - 1)] - final_sorted[:, 0]).tolist(),
        "candidate_final_spread": pairwise_spread(final_emb).tolist(),
        "candidate_traj_spread": pairwise_spread(trajectory_emb).tolist(),
        "topk_final_spread": pairwise_spread(topk_final).tolist(),
        "topk_traj_spread": pairwise_spread(topk_traj).tolist(),
        **topk_emb_dist(final_emb, final_costs, k=5),
    }
    return step_records, final_summary


def mean(xs):
    return float(np.mean(xs))


def aggregate_model(record: dict) -> dict:
    final = record["final"]
    last = record["cem_steps"][-1]
    return {
        "top1_cost_mean": mean(final["top1_cost"]),
        "top2_margin_mean": mean(final["top2_margin"]),
        "cost_std_mean": mean(last["cost_std"]),
        "elite_action_std_mean": mean(last["elite_action_std"]),
        "candidate_traj_spread_mean": mean(final["candidate_traj_spread"]),
        "topk_traj_spread_mean": mean(final["topk_traj_spread"]),
        "top1_top2_emb_l2_mean": mean(final["top1_top2_emb_l2"]),
        "mean_update_norm_last": mean(last["mean_update_norm"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--cem-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.names) != len(args.policies):
        raise ValueError("--names and --policies must have same length")

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
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

    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    action_space = world.envs.action_space
    low = np.asarray(action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

    results = {
        "indices": indices.tolist(),
        "settings": {
            "num_eval": args.num_eval,
            "num_samples": args.num_samples,
            "topk": args.topk,
            "cem_steps": args.cem_steps,
            "seed": args.seed,
            "horizon": int(cfg.plan_config.horizon),
            "action_dim": action_dim,
        },
        "models": {},
    }

    for name, policy in zip(args.names, args.policies):
        print("RUN", name, policy, flush=True)
        cfg.policy = policy
        model = base.load_model(cfg, cache_dir=None)
        step_records, final_summary = trace_cem(
            model,
            prepared_base,
            action_dim=action_dim,
            horizon=int(cfg.plan_config.horizon),
            num_samples=args.num_samples,
            topk=args.topk,
            n_steps=args.cem_steps,
            seed=args.seed,
        )
        record = {
            "policy": policy,
            "cem_steps": step_records,
            "final": final_summary,
        }
        record["aggregate"] = aggregate_model(record)
        results["models"][name] = record
        Path(args.output).write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()

    print(json.dumps({k: v["aggregate"] for k, v in results["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
