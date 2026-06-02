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
    cache_dir = cfg.get("cache_dir", None)
    dataset_path = Path(cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def get_action_scaler(dataset):
    scaler = preprocessing.StandardScaler()
    col_data = dataset.get_col_data("action")
    col_data = col_data[~np.isnan(col_data).any(axis=1)]
    scaler.fit(col_data)
    return scaler


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


def sample_eval_rows(cfg, dataset, seed, num_eval):
    valid_indices = get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=num_eval, replace=False))
    indices = valid_indices[picked]
    rows = dataset.get_row_data(indices)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    return {
        "valid_indices": valid_indices,
        "dataset_indices": indices,
        "episodes": rows[col_name],
        "start_steps": rows["step_idx"],
        "col_name": col_name,
    }


def get_row_sequence(dataset, col_name, episode_id, start_step, length):
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    row_indices = []
    for step in range(int(start_step), int(start_step) + length):
        hit = np.nonzero((episode_idx == episode_id) & (step_idx == step))[0]
        if len(hit) != 1:
            raise RuntimeError(
                f"Expected exactly one row for episode={episode_id} step={step}, got {len(hit)}"
            )
        row_indices.append(int(hit[0]))
    return dataset.get_row_data(np.asarray(row_indices))


def load_model(cfg, cache_dir: str | None):
    model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def make_image_tensor(transform, image):
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    else:
        tensor = image
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return transform(tv_tensors.Image(tensor))


def pack_action_sequence(raw_actions, scaler, horizon, action_block):
    raw_action_dim = raw_actions.shape[-1]
    expected_steps = horizon * action_block
    if raw_actions.shape[0] != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} raw actions, got {raw_actions.shape[0]}"
        )
    scaled = scaler.transform(raw_actions.reshape(-1, raw_action_dim))
    return scaled.reshape(horizon, action_block * raw_action_dim).astype(np.float32)


def sample_random_action_sequence(
    dataset,
    col_name,
    valid_indices,
    scaler,
    horizon,
    action_block,
    rng,
):
    for _ in range(100):
        row_idx = int(rng.choice(valid_indices))
        row = dataset.get_row_data(np.asarray([row_idx]))
        episode_id = row[col_name][0]
        start_step = row["step_idx"][0]
        try:
            seq = get_row_sequence(
                dataset,
                col_name,
                episode_id,
                start_step,
                horizon * action_block,
            )
            return pack_action_sequence(seq["action"], scaler, horizon, action_block)
        except RuntimeError:
            continue
    raise RuntimeError("Failed to sample a valid negative action sequence.")


def build_candidates(
    good_sequence,
    dataset,
    col_name,
    valid_indices,
    scaler,
    horizon,
    action_block,
    noise_scales,
    num_shuffles,
    num_random,
    rng,
):
    candidates = [{"name": "expert", "type": "expert", "actions": good_sequence}]

    for noise_scale in noise_scales:
        noisy = good_sequence + rng.normal(0.0, noise_scale, size=good_sequence.shape)
        candidates.append(
            {
                "name": f"noise_{noise_scale:g}",
                "type": "noise",
                "actions": noisy.astype(np.float32),
            }
        )

    for idx in range(num_shuffles):
        perm = rng.permutation(horizon)
        shuffled = good_sequence[perm].astype(np.float32)
        candidates.append(
            {
                "name": f"shuffle_{idx}",
                "type": "shuffle",
                "actions": shuffled,
            }
        )

    for idx in range(num_random):
        random_seq = sample_random_action_sequence(
            dataset=dataset,
            col_name=col_name,
            valid_indices=valid_indices,
            scaler=scaler,
            horizon=horizon,
            action_block=action_block,
            rng=rng,
        )
        candidates.append(
            {
                "name": f"random_{idx}",
                "type": "random",
                "actions": random_seq,
            }
        )

    return candidates


def score_candidates(
    model,
    transform,
    current_pixels,
    goal_pixels,
    candidates,
):
    num_candidates = len(candidates)
    current = make_image_tensor(transform, current_pixels)[None, None, None, ...]
    goal = make_image_tensor(transform, goal_pixels)[None, None, None, ...]
    info = {
        "pixels": current.expand(1, num_candidates, 1, *current.shape[-3:]).to("cuda"),
        "goal": goal.expand(1, num_candidates, 1, *goal.shape[-3:]).to("cuda"),
        "action": torch.zeros(
            1,
            num_candidates,
            1,
            candidates[0]["actions"].shape[-1],
            dtype=torch.float32,
            device="cuda",
        ),
    }
    action_candidates = torch.tensor(
        np.stack([candidate["actions"] for candidate in candidates], axis=0),
        dtype=torch.float32,
        device="cuda",
    )[None]

    with torch.inference_mode():
        costs = model.get_cost(info, action_candidates)[0].detach().float().cpu().numpy()

    scored = []
    for candidate, cost in zip(candidates, costs.tolist()):
        scored.append(
            {
                "name": candidate["name"],
                "type": candidate["type"],
                "cost": float(cost),
            }
        )
    return scored


def summarize_type(scored_candidates, margin):
    expert_cost = next(item["cost"] for item in scored_candidates if item["type"] == "expert")
    out = {}
    for candidate_type in sorted({item["type"] for item in scored_candidates if item["type"] != "expert"}):
        subset = [item for item in scored_candidates if item["type"] == candidate_type]
        costs = np.asarray([item["cost"] for item in subset], dtype=np.float32)
        best_idx = int(np.argmin(costs))
        best_cost = float(costs[best_idx])
        out[candidate_type] = {
            "count": len(subset),
            "best_name": subset[best_idx]["name"],
            "best_cost": best_cost,
            "best_minus_expert": best_cost - expert_cost,
            "mean_cost": float(costs.mean()),
            "margin_violations": int(np.sum(costs < expert_cost + margin)),
        }
    return out


def build_summary(per_env, margin):
    expert_best = np.asarray([row["expert_is_best"] for row in per_env], dtype=np.float32)
    alt_minus_expert = np.asarray([row["best_alt_minus_expert"] for row in per_env], dtype=np.float32)
    rank_loss = np.asarray([max(0.0, margin - gap) for gap in alt_minus_expert], dtype=np.float32)
    any_bad_better = np.asarray([row["any_bad_better"] for row in per_env], dtype=np.float32)

    by_type = {}
    all_types = sorted(
        {
            candidate_type
            for row in per_env
            for candidate_type in row["type_summary"].keys()
        }
    )
    for candidate_type in all_types:
        gaps = np.asarray(
            [row["type_summary"][candidate_type]["best_minus_expert"] for row in per_env],
            dtype=np.float32,
        )
        violations = np.asarray(
            [row["type_summary"][candidate_type]["margin_violations"] for row in per_env],
            dtype=np.float32,
        )
        by_type[candidate_type] = {
            "best_minus_expert_mean": float(gaps.mean()),
            "best_minus_expert_median": float(np.median(gaps)),
            "margin_violations_mean": float(violations.mean()),
            "expert_beats_type_rate": float(np.mean(gaps > 0.0)),
        }

    return {
        "expert_best_rate": float(expert_best.mean()),
        "any_bad_better_rate": float(any_bad_better.mean()),
        "best_alt_minus_expert_mean": float(alt_minus_expert.mean()),
        "best_alt_minus_expert_median": float(np.median(alt_minus_expert)),
        "rank_loss_mean": float(rank_loss.mean()),
        "rank_loss_median": float(np.median(rank_loss)),
        "by_type": by_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--noise-scales", default="0.25,0.5,1.0")
    parser.add_argument("--num-shuffles", type=int, default=4)
    parser.add_argument("--num-random", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)

    noise_scales = [float(item) for item in args.noise_scales.split(",") if item]
    horizon = int(cfg.plan_config.horizon)
    action_block = int(cfg.plan_config.action_block)
    rng = np.random.default_rng(args.seed)

    dataset = get_dataset(cfg)
    action_scaler = get_action_scaler(dataset)
    transform = img_transform(cfg)
    sampled = sample_eval_rows(cfg, dataset, seed=args.seed, num_eval=args.num_eval)
    model = load_model(cfg, args.cache_dir)

    start_time = time.time()
    per_env = []
    for eval_idx, (episode_id, start_step) in enumerate(
        zip(sampled["episodes"], sampled["start_steps"])
    ):
        sequence = get_row_sequence(
            dataset=dataset,
            col_name=sampled["col_name"],
            episode_id=episode_id,
            start_step=start_step,
            length=horizon * action_block,
        )
        current_row = get_row_sequence(
            dataset=dataset,
            col_name=sampled["col_name"],
            episode_id=episode_id,
            start_step=start_step,
            length=1,
        )
        goal_row = get_row_sequence(
            dataset=dataset,
            col_name=sampled["col_name"],
            episode_id=episode_id,
            start_step=start_step + cfg.eval.goal_offset_steps,
            length=1,
        )

        expert_sequence = pack_action_sequence(
            sequence["action"],
            scaler=action_scaler,
            horizon=horizon,
            action_block=action_block,
        )
        candidates = build_candidates(
            good_sequence=expert_sequence,
            dataset=dataset,
            col_name=sampled["col_name"],
            valid_indices=sampled["valid_indices"],
            scaler=action_scaler,
            horizon=horizon,
            action_block=action_block,
            noise_scales=noise_scales,
            num_shuffles=args.num_shuffles,
            num_random=args.num_random,
            rng=rng,
        )
        scored = score_candidates(
            model=model,
            transform=transform,
            current_pixels=current_row["pixels"][0],
            goal_pixels=goal_row["pixels"][0],
            candidates=candidates,
        )

        expert = next(item for item in scored if item["type"] == "expert")
        alternatives = [item for item in scored if item["type"] != "expert"]
        best_alt = min(alternatives, key=lambda item: item["cost"])
        best_overall = min(scored, key=lambda item: item["cost"])

        per_env.append(
            {
                "eval_index": eval_idx,
                "dataset_index": int(sampled["dataset_indices"][eval_idx]),
                "episode_id": int(episode_id),
                "start_step": int(start_step),
                "expert_cost": float(expert["cost"]),
                "expert_is_best": bool(best_overall["type"] == "expert"),
                "best_overall_name": best_overall["name"],
                "best_overall_type": best_overall["type"],
                "best_overall_cost": float(best_overall["cost"]),
                "best_alt_name": best_alt["name"],
                "best_alt_type": best_alt["type"],
                "best_alt_cost": float(best_alt["cost"]),
                "best_alt_minus_expert": float(best_alt["cost"] - expert["cost"]),
                "any_bad_better": bool(best_alt["cost"] < expert["cost"]),
                "type_summary": summarize_type(scored, margin=args.margin),
                "candidates": scored,
            }
        )

    result = {
        "policy": args.policy,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "elapsed_sec": time.time() - start_time,
        "noise_scales": noise_scales,
        "num_shuffles": args.num_shuffles,
        "num_random": args.num_random,
        "margin": args.margin,
        "summary": build_summary(per_env, margin=args.margin),
        "per_env": per_env,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
