from __future__ import annotations

import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from action_diffusion import ActionDiffusionMLP, DiffusionSchedule


def main() -> None:
    planner_cfg = OmegaConf.load("./config/train/diffusion_planner.yaml")
    data_cfg = OmegaConf.load("./config/train/data/pusht.yaml")
    data_cfg.dataset.num_steps = planner_cfg.planner.horizon
    dataset = swm.data.HDF5Dataset(**data_cfg.dataset, transform=None)
    sample = dataset[0]

    batch = {
        "state": sample["state"][0].unsqueeze(0).float(),
        "goal_state": sample["state"][-1].unsqueeze(0).float(),
        "proprio": sample["proprio"][0].unsqueeze(0).float(),
        "goal_proprio": sample["proprio"][-1].unsqueeze(0).float(),
        "actions": sample["action"][: planner_cfg.planner.horizon].unsqueeze(0).float(),
    }

    model = ActionDiffusionMLP(
        action_dim=planner_cfg.planner.action_dim,
        horizon=planner_cfg.planner.horizon,
        state_dim=planner_cfg.planner.state_dim,
        hidden_dim=planner_cfg.model.hidden_dim,
        depth=planner_cfg.model.depth,
        time_dim=planner_cfg.model.time_dim,
        proprio_dim=planner_cfg.planner.proprio_dim,
    )
    schedule = DiffusionSchedule.cosine(planner_cfg.diffusion.num_steps)
    steps = torch.randint(0, planner_cfg.diffusion.num_steps, (1,))
    noise = torch.randn_like(batch["actions"])
    alpha_bar = schedule.alpha_bars[steps].view(-1, 1, 1)
    noisy_actions = alpha_bar.sqrt() * batch["actions"] + (1 - alpha_bar).sqrt() * noise

    pred = model(
        noisy_actions=noisy_actions,
        timesteps=steps,
        state=batch["state"],
        goal_state=batch["goal_state"],
        proprio=batch["proprio"],
        goal_proprio=batch["goal_proprio"],
    )
    print("SMOKE_OK", pred.shape)


if __name__ == "__main__":
    main()
