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


def mean(xs):
    return float(np.mean(xs))


def pairwise_spread(x: torch.Tensor) -> torch.Tensor:
    vals = []
    x = x.detach().float().cpu()
    for env in x:
        d = torch.cdist(env, env)
        iu = torch.triu_indices(env.shape[0], env.shape[0], offset=1)
        vals.append(d[iu[0], iu[1]].mean())
    return torch.stack(vals)


def add_dual_cost(model, prepared: dict, goal_cost: torch.Tensor, lambda_subspace: float) -> tuple[torch.Tensor, dict]:
    if lambda_subspace == 0.0 or not hasattr(model, "action_subspace_head"):
        zeros = torch.zeros_like(goal_cost)
        return goal_cost, {
            "subspace_anchor_spread": zeros,
            "dual_bonus": zeros,
        }
    pred = prepared["predicted_emb"].float()
    projected = model.action_subspace_head(pred)
    traj = projected.reshape(projected.shape[0], projected.shape[1], -1)
    anchor = traj[:, :1].detach()
    subspace_anchor_spread = (traj - anchor).norm(dim=-1)
    dual_bonus = lambda_subspace * subspace_anchor_spread
    return goal_cost - dual_bonus, {
        "subspace_anchor_spread": subspace_anchor_spread.detach(),
        "dual_bonus": dual_bonus.detach(),
    }


def summarize_step(
    rank_cost: torch.Tensor,
    goal_cost: torch.Tensor,
    candidates: torch.Tensor,
    topk: int,
    extra: dict,
) -> dict[str, list[float]]:
    rank_cpu = rank_cost.detach().float().cpu()
    goal_cpu = goal_cost.detach().float().cpu()
    sorted_rank, topk_idx = torch.topk(rank_cost, k=topk, dim=1, largest=False)
    order = torch.argsort(rank_cost, dim=1)
    sorted_cpu = torch.sort(rank_cpu, dim=1).values
    batch_idx = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
    elite = candidates[batch_idx, topk_idx]
    top1_idx = order[:, 0]
    out = {
        "rank_top1_cost": sorted_cpu[:, 0].tolist(),
        "rank_top2_margin": (sorted_cpu[:, 1] - sorted_cpu[:, 0]).tolist(),
        "rank_cost_std": rank_cpu.std(dim=1).tolist(),
        "goal_top1_cost_by_rank": goal_cpu[torch.arange(goal_cpu.shape[0]), top1_idx.cpu()].tolist(),
        "elite_goal_cost_mean": goal_cpu.gather(1, topk_idx.detach().cpu()).mean(dim=1).tolist(),
        "elite_action_std": elite.detach().float().cpu().reshape(elite.shape[0], elite.shape[1], -1).std(dim=1).mean(dim=1).tolist(),
    }
    if "subspace_anchor_spread" in extra:
        spread = extra["subspace_anchor_spread"].detach().float().cpu()
        bonus = extra["dual_bonus"].detach().float().cpu()
        out["subspace_top1_spread_by_rank"] = spread[torch.arange(spread.shape[0]), top1_idx.cpu()].tolist()
        out["subspace_topk_spread_mean"] = spread.gather(1, topk_idx.detach().cpu()).mean(dim=1).tolist()
        out["dual_top1_bonus_by_rank"] = bonus[torch.arange(bonus.shape[0]), top1_idx.cpu()].tolist()
    return out


@torch.inference_mode()
def trace_dual_cem(
    model,
    prepared_base,
    action_dim: int,
    horizon: int,
    num_samples: int,
    topk: int,
    n_steps: int,
    seed: int,
    lambda_subspace: float,
):
    device = "cuda"
    num_envs = next(v for v in prepared_base.values() if torch.is_tensor(v)).shape[0]
    mean_action = torch.zeros(num_envs, horizon, action_dim, device=device)
    var = torch.ones(num_envs, horizon, action_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    step_records = []
    final_rank_cost = None
    final_goal_cost = None
    final_prepared = None
    final_extra = None

    for step in range(n_steps):
        candidates = torch.randn(
            num_envs,
            num_samples,
            horizon,
            action_dim,
            generator=gen,
            device=device,
        )
        candidates = candidates * var[:, None] + mean_action[:, None]
        candidates[:, 0] = mean_action

        prepared = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in prepared_base.items()
        }
        prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
        goal_cost = model.get_cost(prepared, candidates)
        rank_cost, extra = add_dual_cost(model, prepared, goal_cost, lambda_subspace)

        step_record = summarize_step(rank_cost, goal_cost, candidates, topk=topk, extra=extra)
        step_record["step"] = step
        step_record["mean_action_norm"] = mean_action.detach().float().cpu().reshape(num_envs, -1).norm(dim=1).tolist()
        step_records.append(step_record)

        _, topk_idx = torch.topk(rank_cost, k=topk, dim=1, largest=False)
        batch_idx = torch.arange(num_envs, device=device)[:, None]
        elite = candidates[batch_idx, topk_idx]
        new_mean = elite.mean(dim=1)
        new_var = elite.std(dim=1)
        step_record["mean_update_norm"] = (new_mean - mean_action).detach().float().cpu().reshape(num_envs, -1).norm(dim=1).tolist()
        mean_action, var = new_mean, new_var

        final_rank_cost = rank_cost.detach().cpu()
        final_goal_cost = goal_cost.detach().cpu()
        final_prepared = prepared
        final_extra = {k: v.detach().cpu() for k, v in extra.items()}

    pred = final_prepared["predicted_emb"].detach().float().cpu()
    final_emb = pred[:, :, -1]
    trajectory_emb = pred.reshape(pred.shape[0], pred.shape[1], -1)
    rank_sorted, rank_order = torch.sort(final_rank_cost, dim=1)
    topk_order = rank_order[:, : min(5, rank_order.shape[1])]
    batch_cpu = torch.arange(final_emb.shape[0])[:, None]
    topk_traj = trajectory_emb[batch_cpu, topk_order]
    top1_idx = rank_order[:, 0]
    top2_idx = rank_order[:, 1]
    final_summary = {
        "rank_top1_cost": rank_sorted[:, 0].tolist(),
        "rank_top2_margin": (rank_sorted[:, 1] - rank_sorted[:, 0]).tolist(),
        "goal_top1_cost_by_rank": final_goal_cost[torch.arange(final_goal_cost.shape[0]), top1_idx].tolist(),
        "goal_top2_margin_by_rank": (
            final_goal_cost[torch.arange(final_goal_cost.shape[0]), top2_idx]
            - final_goal_cost[torch.arange(final_goal_cost.shape[0]), top1_idx]
        ).tolist(),
        "candidate_traj_spread": pairwise_spread(trajectory_emb).tolist(),
        "topk_traj_spread": pairwise_spread(topk_traj).tolist(),
    }
    if final_extra and "subspace_anchor_spread" in final_extra:
        spread = final_extra["subspace_anchor_spread"]
        final_summary["subspace_top1_spread_by_rank"] = spread[torch.arange(spread.shape[0]), top1_idx].tolist()
        final_summary["subspace_topk_spread_mean"] = spread.gather(1, topk_order).mean(dim=1).tolist()
    return step_records, final_summary


def aggregate_model(record: dict) -> dict:
    final = record["final"]
    last = record["cem_steps"][-1]
    out = {
        "goal_top1_cost_by_rank_mean": mean(final["goal_top1_cost_by_rank"]),
        "rank_top2_margin_mean": mean(final["rank_top2_margin"]),
        "goal_top2_margin_by_rank_mean": mean(final["goal_top2_margin_by_rank"]),
        "candidate_traj_spread_mean": mean(final["candidate_traj_spread"]),
        "topk_traj_spread_mean": mean(final["topk_traj_spread"]),
        "elite_action_std_mean": mean(last["elite_action_std"]),
        "mean_update_norm_last": mean(last["mean_update_norm"]),
    }
    if "subspace_top1_spread_by_rank" in final:
        out["subspace_top1_spread_by_rank_mean"] = mean(final["subspace_top1_spread_by_rank"])
        out["subspace_topk_spread_mean"] = mean(final["subspace_topk_spread_mean"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, required=True)
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
        "settings": vars(args) | {
            "horizon": int(cfg.plan_config.horizon),
            "action_dim": action_dim,
        },
        "models": {},
    }

    for name, policy in zip(args.names, args.policies):
        print("LOAD", name, policy, flush=True)
        cfg.policy = policy
        model = base.load_model(cfg, cache_dir=None)
        for lam in args.lambdas:
            run_name = f"{name}_lambda{lam:g}"
            print("RUN", run_name, flush=True)
            step_records, final_summary = trace_dual_cem(
                model,
                prepared_base,
                action_dim=action_dim,
                horizon=int(cfg.plan_config.horizon),
                num_samples=args.num_samples,
                topk=args.topk,
                n_steps=args.cem_steps,
                seed=args.seed,
                lambda_subspace=lam,
            )
            record = {
                "policy": policy,
                "lambda_subspace": lam,
                "cem_steps": step_records,
                "final": final_summary,
            }
            record["aggregate"] = aggregate_model(record)
            results["models"][run_name] = record
            Path(args.output).write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()

    print(json.dumps({k: v["aggregate"] for k, v in results["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
