from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_dataset(cfg):
    dataset_path = Path(cfg.get("cache_dir", swm.data.utils.get_cache_dir()))
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def get_process(cfg, dataset):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def get_valid_indices(cfg, dataset):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    episode_len = []
    for ep_id in ep_indices:
        episode_len.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    episode_len = np.asarray(episode_len)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.asarray([max_start_idx_dict[ep_id] for ep_id in episode_idx])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    return np.nonzero(valid_mask)[0]


def build_info_dict(cfg, dataset, indices):
    rows = dataset.get_row_data(indices)
    goal_rows = dataset.get_row_data(indices + cfg.eval.goal_offset_steps)
    return {
        "pixels": rows["pixels"][:, None, ...],
        "goal": goal_rows["pixels"][:, None, ...],
        "action": rows["action"][:, None, ...],
        "proprio": rows["proprio"][:, None, ...],
        "goal_proprio": goal_rows["proprio"][:, None, ...],
        "state": rows["state"][:, None, ...],
        "goal_state": goal_rows["state"][:, None, ...],
    }


def make_eval_like_info(raw_info, transform, process):
    prepared = {}
    for k, v in raw_info.items():
        is_numpy = isinstance(v, (np.ndarray, np.generic))
        if k in process:
            shape = v.shape
            if len(shape) > 2:
                v = v.reshape(-1, *shape[2:])
            v = process[k].transform(v)
            v = v.reshape(shape)
        if k in transform:
            shape = None
            if is_numpy or torch.is_tensor(v):
                if v.ndim > 2:
                    shape = v.shape
                    v = v.reshape(-1, *shape[2:])
            if k.startswith("pixels") or k.startswith("goal"):
                if is_numpy:
                    v = np.transpose(v, (0, 3, 1, 2))
                else:
                    v = v.permute(0, 3, 1, 2)
            v = torch.stack([transform[k](tv_tensors.Image(x)) for x in v])
            is_numpy = False
            if shape is not None:
                v = v.reshape(*shape[:2], *v.shape[1:])
        if is_numpy and v.dtype.kind not in "USO":
            v = torch.from_numpy(v)
        prepared[k] = v
    return prepared


def expand_info_for_candidates(info_dict, num_envs, num_samples):
    expanded = {}
    for k, v in info_dict.items():
        if torch.is_tensor(v):
            expanded[k] = v.unsqueeze(1).expand(num_envs, num_samples, *v.shape[1:])
        else:
            expanded[k] = v
    return expanded


def sample_candidates(action_space, n_envs, num_candidates, horizon, action_block, seed):
    rng = np.random.default_rng(seed)
    low = np.asarray(action_space.low)
    high = np.asarray(action_space.high)
    if low.ndim > 1:
        low = low[0]
        high = high[0]
    low = np.tile(low.reshape(-1), action_block)
    high = np.tile(high.reshape(-1), action_block)
    samples = rng.uniform(
        low=low,
        high=high,
        size=(n_envs, num_candidates, horizon, low.shape[0]),
    )
    return torch.tensor(samples, dtype=torch.float32)


def tensor_stats(x):
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "norm_mean": float(x.norm(dim=-1).mean().item()),
        "norm_std": float(x.norm(dim=-1).std().item()),
    }


@torch.no_grad()
def trace_rollout(model, info, action_sequence, history_size=3):
    device = next(model.parameters()).device
    info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
    action_sequence = action_sequence.to(device)

    goal = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]
    for k in list(info.keys()):
        if k.startswith("goal_"):
            goal[k[len("goal_") :]] = goal.pop(k)
    goal.pop("action")
    goal = model.encode(goal)
    goal_emb = goal["emb"]

    h = info["pixels"].size(2)
    b, s, t = action_sequence.shape[:3]
    act_0, act_future = torch.split(action_sequence, [h, t - h], dim=2)
    info["action"] = act_0

    init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
    init = model.encode(init)
    emb0 = init["emb"]
    emb = emb0.unsqueeze(1).expand(b, s, -1, -1)
    emb = rearrange(emb, "b s ... -> (b s) ...").clone()
    act = rearrange(act_0, "b s ... -> (b s) ...")
    act_future = rearrange(act_future, "b s ... -> (b s) ...")

    step_embs = []
    hs = history_size
    for step in range(t - h):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -hs:], act_emb[:, -hs:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        step_embs.append(rearrange(pred.squeeze(1), "(b s) d -> b s d", b=b, s=s))
        act = torch.cat([act, act_future[:, step : step + 1]], dim=1)

    act_emb = model.action_encoder(act)
    pred = model.predict(emb[:, -hs:], act_emb[:, -hs:])[:, -1:]
    emb = torch.cat([emb, pred], dim=1)
    step_embs.append(rearrange(pred.squeeze(1), "(b s) d -> b s d", b=b, s=s))
    pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=b, s=s)

    goal_last = goal_emb[:, -1, :].unsqueeze(1)
    final = pred_rollout[:, :, -1, :]
    costs = F.mse_loss(final, goal_last.detach().expand_as(final), reduction="none").sum(dim=-1)
    sorted_costs, sorted_idx = torch.sort(costs, dim=1)

    per_step = []
    for i, x in enumerate(step_embs, start=1):
        goal_dist = (x - goal_last).pow(2).sum(dim=-1).sqrt()
        spread = x.var(dim=1).sum(dim=-1).sqrt()
        top1_x = x.gather(1, sorted_idx[:, :1, None].expand(-1, -1, x.shape[-1])).squeeze(1)
        top1_goal = (top1_x - goal_emb[:, -1, :]).pow(2).sum(dim=-1).sqrt()
        per_step.append(
            {
                "step": i,
                "pred_norm_mean": float(x.norm(dim=-1).mean().item()),
                "candidate_spread_mean": float(spread.mean().item()),
                "goal_dist_mean": float(goal_dist.mean().item()),
                "top1_goal_dist_mean": float(top1_goal.mean().item()),
            }
        )

    return {
        "init_emb": emb0.detach().cpu(),
        "goal_emb": goal_emb.detach().cpu(),
        "costs": costs.detach().cpu(),
        "sorted_costs": sorted_costs.detach().cpu(),
        "per_step": per_step,
    }


def load_model(policy, cache_dir):
    model = swm.policy.AutoCostModel(policy, cache_dir=cache_dir).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.cache_dir = "/data1/jingyixi/.stable_worldmodel"
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    dataset = get_dataset(cfg)
    process = get_process(cfg, dataset)
    valid_indices = get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]
    raw_info = build_info_dict(cfg, dataset, indices)
    prepared = make_eval_like_info(raw_info, {"pixels": img_transform(cfg), "goal": img_transform(cfg)}, process)
    info = expand_info_for_candidates(prepared, args.num_eval, args.num_candidates)
    action_space = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224)).envs.action_space
    candidates = sample_candidates(
        action_space,
        args.num_eval,
        args.num_candidates,
        cfg.plan_config.horizon,
        cfg.plan_config.action_block,
        args.seed,
    )

    policies = {
        "official_clean_ep13": "pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13",
        "pred6_ep7": "pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7",
        "gate07_ep4": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4",
    }

    out = {"seed": args.seed, "num_eval": args.num_eval, "num_candidates": args.num_candidates, "models": {}}
    for name, policy in policies.items():
        print(f"RUN {name}", flush=True)
        model = load_model(policy, cfg.cache_dir)
        tr = trace_rollout(model, {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()}, candidates)
        costs = tr["costs"]
        sorted_costs = tr["sorted_costs"]
        init = tr["init_emb"]
        goal = tr["goal_emb"]
        init_goal = (init[:, -1, :] - goal[:, -1, :]).pow(2).sum(dim=-1).sqrt()
        out["models"][name] = {
            "init_emb": tensor_stats(init.reshape(-1, init.shape[-1])),
            "goal_emb": tensor_stats(goal.reshape(-1, goal.shape[-1])),
            "init_goal_dist_mean": float(init_goal.mean().item()),
            "cost_top1_mean": float(sorted_costs[:, 0].mean().item()),
            "cost_top2_margin_mean": float((sorted_costs[:, 1] - sorted_costs[:, 0]).mean().item()),
            "cost_top5_margin_mean": float((sorted_costs[:, 4] - sorted_costs[:, 0]).mean().item()),
            "cost_std_mean": float(costs.std(dim=1).mean().item()),
            "per_step": tr["per_step"],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        del model
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
