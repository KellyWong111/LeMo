from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))

import analyze_cem_margin as base


class FixedPlanPolicy:
    def __init__(self, plans: np.ndarray):
        self.plans = plans.astype(np.float32)
        self.action_buffer = None

    def set_env(self, env):
        self.env = env
        plans = self.plans.reshape(self.plans.shape[0], -1, self.plans.shape[-1])
        self.action_buffer = deque(plans.transpose(1, 0, 2), maxlen=plans.shape[1])

    def get_action(self, info_dict, **kwargs):
        if self.action_buffer and len(self.action_buffer) > 0:
            return self.action_buffer.popleft()
        return np.zeros((self.env.num_envs, self.plans.shape[-1]), dtype=np.float32)


def load_plan_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "indices": z["indices"],
        "actions": z["actions"].astype(np.float32),
        "costs": z["costs"].astype(np.float32),
        "labels": z["labels"].astype(bool),
    }


def prepare_eval_start(cfg, dataset, indices):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    rows = dataset.get_row_data(indices)
    eval_episodes = rows[col_name]
    eval_start_idx = rows["step_idx"]
    end_steps = np.asarray(eval_start_idx) + int(cfg.eval.goal_offset_steps)
    data = dataset.load_chunk(np.asarray(eval_episodes), np.asarray(eval_start_idx), end_steps)
    columns = dataset.column_names
    init_step_per_env = defaultdict(list)
    goal_step_per_env = defaultdict(list)
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue
            init_data = ep[col][0]
            goal_data = ep[col][-1]
            if not isinstance(init_data, (np.ndarray, torch.Tensor)):
                continue
            if isinstance(init_data, torch.Tensor):
                init_data = init_data.numpy()
            if isinstance(goal_data, torch.Tensor):
                goal_data = goal_data.numpy()
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)
    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[key] = np.stack(value)
    return eval_episodes, eval_start_idx, init_step, goal_step


def reset_world_like_eval(world, cfg, init_step, goal_step):
    seeds = init_step.get("seed")
    init = deepcopy(init_step)
    init.update(deepcopy(goal_step))
    options = [{} for _ in range(world.num_envs)]
    world.reset(seed=seeds, options=options)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) or []
    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                if args_data.get("in_dataset", True):
                    if value in init:
                        prepared_args[args_name] = deepcopy(init[value][i])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_hist = {k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in init.items()}
    goal_hist = {k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in goal_step.items()}
    world.infos.update(deepcopy(init_hist))
    world.infos.update(deepcopy(goal_hist))


def inverse_actions_if_needed(cfg, process, world, plans):
    action_shape = tuple(world.envs.action_space.shape)
    env_action_dim = int(action_shape[-1])
    plans = plans.reshape(plans.shape[0], plans.shape[1], int(cfg.plan_config.action_block), env_action_dim)
    if "action" in process:
        flat = plans.reshape(-1, env_action_dim)
        flat = process["action"].inverse_transform(flat)
        plans = flat.reshape(plans.shape)
    return plans.astype(np.float32)


def angle_abs_err(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def collect_from_infos(infos):
    state = np.asarray(infos["state"][:, -1], dtype=np.float32)
    goal_pose = np.asarray(infos.get("goal_pose", infos.get("goal_state"))[:, -1], dtype=np.float32)
    agent_xy = state[:, 0:2]
    obj_xy = state[:, 2:4]
    obj_angle = state[:, 4]
    if goal_pose.shape[-1] >= 3:
        goal_xy = goal_pose[:, 0:2]
        goal_angle = goal_pose[:, 2]
    else:
        goal_state = np.asarray(infos["goal_state"][:, -1], dtype=np.float32)
        goal_xy = goal_state[:, 2:4]
        goal_angle = goal_state[:, 4]
    contacts = np.asarray(infos.get("n_contacts", np.zeros((state.shape[0], 1)))[:, -1], dtype=np.float32)
    return agent_xy, obj_xy, obj_angle, goal_xy, goal_angle, contacts


def replay_rank(cfg, dataset, process, indices, plans_rank, rank, out_arrays):
    _, _, init_step, goal_step = prepare_eval_start(cfg, dataset, indices)
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224), verbose=0)
    plans_env = inverse_actions_if_needed(cfg, process, world, plans_rank)
    policy = FixedPlanPolicy(plans_env)
    world.set_policy(policy)
    reset_world_like_eval(world, cfg, init_step, goal_step)

    n_env = plans_rank.shape[0]
    eval_budget = int(cfg.eval.eval_budget)
    agent = np.zeros((n_env, eval_budget + 1, 2), dtype=np.float32)
    obj = np.zeros((n_env, eval_budget + 1, 2), dtype=np.float32)
    angle = np.zeros((n_env, eval_budget + 1), dtype=np.float32)
    goal_xy = np.zeros((n_env, 2), dtype=np.float32)
    goal_angle = np.zeros((n_env,), dtype=np.float32)
    contacts = np.zeros((n_env, eval_budget + 1), dtype=np.float32)
    success = np.zeros((n_env,), dtype=bool)
    a, o, th, gxy, ga, ct = collect_from_infos(world.infos)
    agent[:, 0] = a
    obj[:, 0] = o
    angle[:, 0] = th
    goal_xy[:] = gxy
    goal_angle[:] = ga
    contacts[:, 0] = ct

    for t in range(eval_budget):
        world.infos.update(deepcopy(goal_step))
        world.step()
        success = np.logical_or(success, world.terminateds)
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
        a, o, th, gxy, ga, ct = collect_from_infos(world.infos)
        agent[:, t + 1] = a
        obj[:, t + 1] = o
        angle[:, t + 1] = th
        contacts[:, t + 1] = ct

    dist = np.linalg.norm(obj - goal_xy[:, None, :], axis=-1).astype(np.float32)
    progress_curve = (dist[:, :1] - dist).astype(np.float32)
    angle_err = angle_abs_err(angle, goal_angle[:, None]).astype(np.float32)
    obj_step = np.diff(obj, axis=1)
    action_flat = plans_env.reshape(n_env, -1, plans_env.shape[-1])
    action_step = np.diff(action_flat, axis=1) if action_flat.shape[1] > 1 else np.zeros((n_env, 1, action_flat.shape[-1]), dtype=np.float32)
    smoothness = np.sqrt((action_step**2).sum(axis=-1)).mean(axis=1).astype(np.float32)
    obj_path_len = np.sqrt((obj_step**2).sum(axis=-1)).sum(axis=1).astype(np.float32)
    proximity = np.linalg.norm(agent - obj, axis=-1).min(axis=1).astype(np.float32)

    out_arrays["agent_xy"][:, rank] = agent
    out_arrays["object_xy"][:, rank] = obj
    out_arrays["object_angle"][:, rank] = angle
    out_arrays["goal_xy"][:, rank] = goal_xy
    out_arrays["goal_angle"][:, rank] = goal_angle
    out_arrays["distance_curve"][:, rank] = dist
    out_arrays["angle_error_curve"][:, rank] = angle_err
    out_arrays["progress_curve"][:, rank] = progress_curve
    out_arrays["contact_curve"][:, rank] = contacts
    out_arrays["final_distance"][:, rank] = dist[:, -1]
    out_arrays["final_angle_error"][:, rank] = angle_err[:, -1]
    out_arrays["min_distance"][:, rank] = dist.min(axis=1)
    out_arrays["final_progress"][:, rank] = progress_curve[:, -1]
    out_arrays["max_progress"][:, rank] = progress_curve.max(axis=1)
    out_arrays["trajectory_smoothness"][:, rank] = smoothness
    out_arrays["object_path_len"][:, rank] = obj_path_len
    out_arrays["contact_proxy"][:, rank] = contacts.max(axis=1)
    out_arrays["agent_object_min_dist"][:, rank] = proximity
    out_arrays["replay_success"][:, rank] = success
    world.close()


def dump_variant(cfg, dataset, process, seed: int, variant: str, in_path: Path, out_path: Path, ranks: list[int] | None):
    data = load_plan_npz(in_path)
    indices = data["indices"]
    actions = data["actions"]
    labels = data["labels"]
    n_ep, topk = labels.shape
    eval_budget = int(cfg.eval.eval_budget)
    ranks = ranks if ranks is not None else list(range(topk))
    arrays = {
        "agent_xy": np.zeros((n_ep, topk, eval_budget + 1, 2), dtype=np.float32),
        "object_xy": np.zeros((n_ep, topk, eval_budget + 1, 2), dtype=np.float32),
        "object_angle": np.zeros((n_ep, topk, eval_budget + 1), dtype=np.float32),
        "goal_xy": np.zeros((n_ep, topk, 2), dtype=np.float32),
        "goal_angle": np.zeros((n_ep, topk), dtype=np.float32),
        "distance_curve": np.zeros((n_ep, topk, eval_budget + 1), dtype=np.float32),
        "angle_error_curve": np.zeros((n_ep, topk, eval_budget + 1), dtype=np.float32),
        "progress_curve": np.zeros((n_ep, topk, eval_budget + 1), dtype=np.float32),
        "contact_curve": np.zeros((n_ep, topk, eval_budget + 1), dtype=np.float32),
        "final_distance": np.zeros((n_ep, topk), dtype=np.float32),
        "final_angle_error": np.zeros((n_ep, topk), dtype=np.float32),
        "min_distance": np.zeros((n_ep, topk), dtype=np.float32),
        "final_progress": np.zeros((n_ep, topk), dtype=np.float32),
        "max_progress": np.zeros((n_ep, topk), dtype=np.float32),
        "trajectory_smoothness": np.zeros((n_ep, topk), dtype=np.float32),
        "object_path_len": np.zeros((n_ep, topk), dtype=np.float32),
        "contact_proxy": np.zeros((n_ep, topk), dtype=np.float32),
        "agent_object_min_dist": np.zeros((n_ep, topk), dtype=np.float32),
        "replay_success": np.zeros((n_ep, topk), dtype=bool),
    }
    for r in ranks:
        print(f"[REPLAY] variant={variant} seed={seed} rank={r}", flush=True)
        replay_rank(cfg, dataset, process, indices, actions[:, r], r, arrays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        variant=variant,
        seed=int(seed),
        indices=indices,
        costs=data["costs"],
        labels=labels,
        actions=actions,
        **arrays,
    )
    match = float((arrays["replay_success"][:, ranks] == labels[:, ranks]).mean() * 100.0)
    rec = {"variant": variant, "seed": int(seed), "episodes": int(n_ep), "ranks": ranks, "label_match_pct": match}
    print("[DONE]", json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/env_traj_features_n100")
    ap.add_argument("--bsl-pool", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--st-pool", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--variants", default="baseline,vf05_mix20")
    ap.add_argument("--ranks", default="")
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--receding-horizon", type=int, default=4)
    args = ap.parse_args()

    cfg = OmegaConf.load(str(REPO / "config/eval/pusht.yaml"))
    cfg.policy = "baseline"
    cfg.eval.num_eval = args.num_eval
    cfg.cache_dir = os.environ.get("STABLEWM_HOME", "/data1/jingyixi/.stable_worldmodel")
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    OmegaConf.update(cfg, "plan_config.horizon", args.horizon, merge=True)
    OmegaConf.update(cfg, "plan_config.action_block", args.action_block, merge=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", args.receding_horizon, merge=True)
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    ranks = [int(x) for x in args.ranks.split(",") if x != ""] if args.ranks else None
    rows = []
    out = Path(args.output_dir)
    for seed in [int(x) for x in args.seeds.split(",") if x]:
        for variant in [x for x in args.variants.split(",") if x]:
            pool = Path(args.bsl_pool if variant == "baseline" else args.st_pool)
            in_path = pool / f"{variant}_seed{seed}.npz"
            out_path = out / f"{variant}_seed{seed}.npz"
            rows.append(dump_variant(cfg, dataset, process, seed, variant, in_path, out_path, ranks))
            (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
