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


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    flat = x.reshape(-1, x.shape[-1])
    norms = flat.norm(dim=-1)
    out = {
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()),
    }
    if flat.shape[0] >= 3:
        cov = torch.cov(flat.T)
        eig = torch.clamp(torch.linalg.eigvalsh(cov), min=1e-12)
        out["cov_trace"] = float(eig.sum().item())
        out["participation_ratio"] = float(((eig.sum() ** 2) / eig.square().sum()).item())
    return out


def pairwise_candidate_spread(x: torch.Tensor) -> torch.Tensor:
    # x: (B, S, D)
    vals = []
    for env in x.detach().float().cpu():
        d = torch.cdist(env, env)
        iu = torch.triu_indices(env.shape[0], env.shape[0], offset=1)
        vals.append(d[iu[0], iu[1]].mean())
    return torch.stack(vals)


def topk_activation_distance(final_emb: torch.Tensor, costs: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(costs.detach().cpu(), dim=1)
    final = final_emb.detach().float().cpu()
    top1 = final[torch.arange(final.shape[0]), order[:, 0]]
    top2 = final[torch.arange(final.shape[0]), order[:, 1]]
    top5 = final[torch.arange(final.shape[0]), order[:, 4]]
    return {
        "top1_top2_emb_l2_mean": float((top2 - top1).norm(dim=-1).mean().item()),
        "top1_top5_emb_l2_mean": float((top5 - top1).norm(dim=-1).mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.policies) != len(args.names):
        raise ValueError("--policies and --names must have same length")

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
    action_space = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224)).envs.action_space
    candidates = base.sample_candidates(
        action_space,
        n_envs=args.num_eval,
        num_candidates=args.num_candidates,
        horizon=cfg.plan_config.horizon,
        action_block=cfg.plan_config.action_block,
        seed=args.seed,
    ).to("cuda")

    results = {}
    for name, policy in zip(args.names, args.policies):
        print("RUN", name, policy, flush=True)
        cfg.policy = policy
        model = base.load_model(cfg, cache_dir=None)
        prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
        prepared = base.expand_info_for_candidates(prepared, args.num_eval, args.num_candidates)
        cost = model.get_cost(prepared, candidates).detach().cpu()
        pred = prepared["predicted_emb"].detach().float().cpu()  # B,S,T,D
        final = pred[:, :, -1]
        spread_final = pairwise_candidate_spread(final)
        spread_all = pairwise_candidate_spread(pred.reshape(pred.shape[0], pred.shape[1], -1))
        sorted_costs, _ = torch.sort(cost, dim=1)
        result = {
            "policy": policy,
            "num_eval": args.num_eval,
            "num_candidates": args.num_candidates,
            "top1_cost_mean": float(sorted_costs[:, 0].mean().item()),
            "top2_margin_mean": float((sorted_costs[:, 1] - sorted_costs[:, 0]).mean().item()),
            "top5_margin_mean": float((sorted_costs[:, 4] - sorted_costs[:, 0]).mean().item()),
            "final_candidate_spread_mean": float(spread_final.mean().item()),
            "trajectory_candidate_spread_mean": float(spread_all.mean().item()),
            "per_env_final_spread": [float(x) for x in spread_final.tolist()],
            "per_env_top2_margin": [float(x) for x in (sorted_costs[:, 1] - sorted_costs[:, 0]).tolist()],
            **{f"final_{k}": v for k, v in tensor_stats(final).items()},
            **topk_activation_distance(final, cost),
        }
        results[name] = result
        Path(args.output).write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
