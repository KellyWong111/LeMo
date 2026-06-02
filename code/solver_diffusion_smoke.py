from __future__ import annotations

import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def main() -> None:
    eval_cfg = OmegaConf.load("./config/eval/pusht.yaml")
    solver_cfg = OmegaConf.load("./config/eval/solver/diffusion.yaml")
    eval_cfg.policy = (
        "/home/jingyixi/.stable_worldmodel/"
        "pusht_encoder_moda_v14_full_visible_bs32/"
        "lewm_encoder_moda_v14_full_visible_bs32_epoch_4"
    )
    solver_cfg.device = "cuda"
    solver_cfg.planner_ckpt = (
        "/data1/jingyixi/.stable_worldmodel/"
        "pusht_action_diffusion_planner/"
        "pusht_action_diffusion_planner_best.ckpt"
    )
    solver_cfg.rerank_topk = 4

    transform = {
        "pixels": img_transform(eval_cfg.eval.img_size),
        "goal": img_transform(eval_cfg.eval.img_size),
    }

    dataset = swm.data.HDF5Dataset(
        eval_cfg.eval.dataset_name,
        keys_to_cache=eval_cfg.dataset.keys_to_cache,
        cache_dir=Path(swm.data.utils.get_cache_dir()),
    )

    process = {}
    for col in eval_cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    row = dataset.get_row_data([0])
    goal_row = dataset.get_row_data([int(eval_cfg.eval.goal_offset_steps)])
    info = {
        "pixels": row["pixels"],
        "goal": goal_row["pixels"],
        "action": row["action"],
        "proprio": row["proprio"],
        "goal_proprio": goal_row["proprio"],
        "state": row["state"],
        "goal_state": goal_row["state"],
    }

    world = swm.World(
        env_name=eval_cfg.world.env_name,
        num_envs=1,
        max_episode_steps=10,
        history_size=1,
        frame_skip=1,
        image_shape=(224, 224),
    )
    model = swm.policy.AutoCostModel(eval_cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_cfg = swm.PlanConfig(**eval_cfg.plan_config)
    solver = hydra.utils.instantiate(solver_cfg, model=model)
    solver.configure(action_space=world.envs.action_space, n_envs=1, config=plan_cfg)

    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
    )
    policy.set_env(world.envs)
    prepared = policy._prepare_info(info)
    print(
        "PREPARED",
        {k: tuple(v.shape) if torch.is_tensor(v) else type(v) for k, v in prepared.items()},
        flush=True,
    )
    outputs = solver(prepared, init_action=None)
    print("SMOKE_SOLVE_OK", tuple(outputs["actions"].shape), outputs.get("stats"), flush=True)


if __name__ == "__main__":
    main()
