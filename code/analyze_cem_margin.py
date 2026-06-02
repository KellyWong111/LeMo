from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors


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
    cache_dir = cfg.get("cache_dir", None)
    dataset_path = Path(cache_dir or swm.data.utils.get_cache_dir())
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
    episode_len = np.array(episode_len)

    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    return np.nonzero(valid_mask)[0]


def build_info_dict(cfg, dataset, process, indices):
    rows = dataset.get_row_data(indices)
    goal_rows = dataset.get_row_data(indices + cfg.eval.goal_offset_steps)

    info = {
        "pixels": rows["pixels"][:, None, ...],
        "goal": goal_rows["pixels"][:, None, ...],
        "action": rows["action"][:, None, ...],
        "proprio": rows["proprio"][:, None, ...],
        "goal_proprio": goal_rows["proprio"][:, None, ...],
        "state": rows["state"][:, None, ...],
        "goal_state": goal_rows["state"][:, None, ...],
    }

    return info


def sample_candidates(action_space, n_envs, num_candidates, horizon, action_block, seed):
    rng = np.random.default_rng(seed)
    low = np.asarray(action_space.low)
    high = np.asarray(action_space.high)
    if low.ndim > 1:
        low = low[0]
        high = high[0]
    low = np.tile(low.reshape(-1), action_block)
    high = np.tile(high.reshape(-1), action_block)
    action_dim = low.shape[0]
    samples = rng.uniform(
        low=low,
        high=high,
        size=(n_envs, num_candidates, horizon, action_dim),
    )
    return torch.tensor(samples, dtype=torch.float32)


def load_model(cfg, cache_dir: str | None):
    model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def expand_info_for_candidates(info_dict: dict[str, torch.Tensor], num_envs: int, num_samples: int):
    expanded = {}
    for k, v in info_dict.items():
        if not torch.is_tensor(v):
            expanded[k] = v
            continue
        if v.ndim >= 3:
            expanded[k] = v.unsqueeze(1).expand(num_envs, num_samples, *v.shape[1:])
        else:
            expanded[k] = v.unsqueeze(1).expand(num_envs, num_samples, *v.shape[1:])
    return expanded


def make_eval_like_info(raw_info, transform, process):
    prepared = {}
    for k, v in raw_info.items():
        is_numpy = isinstance(v, (np.ndarray, np.generic))

        if k in process:
            if not is_numpy:
                raise ValueError(f"Expected numpy array for key '{k}' in process, got {type(v)}")
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
            is_numpy = isinstance(v, (np.ndarray | np.generic))
            if shape is not None:
                v = v.reshape(*shape[:2], *v.shape[1:])

        if is_numpy and v.dtype.kind not in "USO":
            v = torch.from_numpy(v)

        prepared[k] = v
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)

    dataset = get_dataset(cfg)
    process = get_process(cfg, dataset)
    valid_indices = get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]

    raw_info = build_info_dict(cfg, dataset, process, indices)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    prepared = make_eval_like_info(raw_info, transform, process)

    model = load_model(cfg, args.cache_dir)
    prepared = expand_info_for_candidates(prepared, args.num_eval, args.num_candidates)
    action_space = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224)).envs.action_space
    candidates = sample_candidates(
        action_space,
        n_envs=args.num_eval,
        num_candidates=args.num_candidates,
        horizon=cfg.plan_config.horizon,
        action_block=cfg.plan_config.action_block,
        seed=args.seed,
    ).to("cuda")

    start = time.time()
    costs = model.get_cost(prepared, candidates).detach().cpu()
    elapsed = time.time() - start
    sorted_costs, _ = torch.sort(costs, dim=1)
    top1 = sorted_costs[:, 0]
    top2 = sorted_costs[:, 1]
    top5 = sorted_costs[:, 4] if sorted_costs.shape[1] >= 5 else sorted_costs[:, -1]

    result = {
        "policy": args.policy,
        "num_eval": args.num_eval,
        "num_candidates": args.num_candidates,
        "elapsed_sec": elapsed,
        "top1_mean": float(top1.mean().item()),
        "top2_mean": float(top2.mean().item()),
        "top5_mean": float(top5.mean().item()),
        "top2_margin_mean": float((top2 - top1).mean().item()),
        "top5_margin_mean": float((top5 - top1).mean().item()),
        "per_env_top1": [float(x) for x in top1.tolist()],
        "per_env_top2": [float(x) for x in top2.tolist()],
        "per_env_top2_margin": [float(x) for x in (top2 - top1).tolist()],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
