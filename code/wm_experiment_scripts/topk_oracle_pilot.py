from __future__ import annotations

import argparse
import json
import os
from collections import deque
from copy import deepcopy
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base


class FixedPlanPolicy:
    def __init__(self, plans: np.ndarray, action_block: int):
        self.plans = plans
        self.action_block = action_block
        self.action_buffer = None

    def set_env(self, env):
        self.env = env
        plans = self.plans.reshape(
            self.plans.shape[0],
            -1,
            self.plans.shape[-1],
        )
        self.action_buffer = deque(
            plans.transpose(1, 0, 2),
            maxlen=plans.shape[1],
        )

    def get_action(self, info_dict, **kwargs):
        if self.action_buffer and len(self.action_buffer) > 0:
            return self.action_buffer.popleft()
        return np.zeros((self.env.num_envs, self.plans.shape[-1]), dtype=np.float32)


@torch.inference_mode()
def get_topk_candidates(model, prepared_base, action_dim, horizon, num_samples, topk, n_steps, seed):
    device = "cuda"
    num_envs = next(v for v in prepared_base.values() if torch.is_tensor(v)).shape[0]
    mean = torch.zeros(num_envs, horizon, action_dim, device=device)
    var = torch.ones(num_envs, horizon, action_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    final_costs = None
    final_candidates = None
    for _ in range(n_steps):
        candidates = torch.randn(num_envs, num_samples, horizon, action_dim, generator=gen, device=device)
        candidates = candidates * var[:, None] + mean[:, None]
        candidates[:, 0] = mean
        prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
        prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
        costs = model.get_cost(prepared, candidates)
        _, idx = torch.topk(costs, k=topk, dim=1, largest=False)
        batch = torch.arange(num_envs, device=device)[:, None]
        elite = candidates[batch, idx]
        mean = elite.mean(dim=1)
        var = elite.std(dim=1)
        final_costs = costs.detach().cpu()
        final_candidates = candidates.detach().cpu()
    order = torch.argsort(final_costs, dim=1)[:, :topk]
    batch_cpu = torch.arange(final_candidates.shape[0])[:, None]
    topk_candidates = final_candidates[batch_cpu, order]
    topk_costs = final_costs[batch_cpu, order]
    return topk_candidates, topk_costs


@torch.inference_mode()
def get_multistart_topk_candidates(
    model,
    prepared_base,
    action_dim,
    horizon,
    num_samples,
    topk,
    n_steps,
    seed,
    restarts,
):
    all_candidates = []
    all_costs = []
    for restart in range(restarts):
        candidates, costs = get_topk_candidates(
            model,
            prepared_base,
            action_dim=action_dim,
            horizon=horizon,
            num_samples=num_samples,
            topk=topk,
            n_steps=n_steps,
            seed=seed + restart * 1009,
        )
        all_candidates.append(candidates)
        all_costs.append(costs)
    candidates = torch.cat(all_candidates, dim=1)
    costs = torch.cat(all_costs, dim=1)
    order = torch.argsort(costs, dim=1)[:, :topk]
    batch = torch.arange(candidates.shape[0])[:, None]
    return candidates[batch, order], costs[batch, order]


def eval_fixed_plans(cfg, dataset, process, indices, plans, save_video=False):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    rows = dataset.get_row_data(indices)
    eval_episodes = rows[col_name]
    eval_start_idx = rows["step_idx"]
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    action_shape = tuple(world.envs.action_space.shape)
    env_action_dim = int(action_shape[-1])
    if "action" in process:
        plans = plans.reshape(
            plans.shape[0],
            plans.shape[1],
            int(cfg.plan_config.action_block),
            env_action_dim,
        )
        flat = plans.reshape(-1, env_action_dim)
        flat = process["action"].inverse_transform(flat)
        plans = flat.reshape(plans.shape)
    else:
        plans = plans.reshape(
            plans.shape[0],
            plans.shape[1],
            int(cfg.plan_config.action_block),
            env_action_dim,
        )
    policy = FixedPlanPolicy(plans.astype(np.float32), action_block=int(cfg.plan_config.action_block))
    world.set_policy(policy)
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=save_video,
        video_path="/tmp/topk_oracle_videos",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=1)
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
    topk_candidates, topk_costs = get_multistart_topk_candidates(
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
    del model
    torch.cuda.empty_cache()

    plans = topk_candidates.numpy()
    top1_metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, 0])
    oracle_success = []
    candidate_successes = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        succ = np.asarray(metrics["episode_successes"], dtype=bool)
        candidate_successes.append(succ.tolist())
        oracle_success.append(succ)
    oracle_success = np.stack(oracle_success, axis=1)
    oracle_any = oracle_success.any(axis=1)
    first_success_rank = []
    for row in oracle_success:
        hits = np.nonzero(row)[0]
        first_success_rank.append(int(hits[0]) if len(hits) else None)

    result = {
        "policy": args.policy,
        "indices": indices.tolist(),
        "settings": vars(args),
        "top1_success_rate": float(np.mean(top1_metrics["episode_successes"]) * 100.0),
        "oracle_topk_success_rate": float(np.mean(oracle_any) * 100.0),
        "top1_episode_successes": np.asarray(top1_metrics["episode_successes"], dtype=bool).tolist(),
        "oracle_episode_successes": oracle_any.tolist(),
        "first_success_rank": first_success_rank,
        "candidate_successes_by_rank": candidate_successes,
        "topk_costs": topk_costs.tolist(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
