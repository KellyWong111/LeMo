from __future__ import annotations

from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from action_diffusion import ActionDiffusionMLP, DiffusionSchedule


class PlannerSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, horizon: int) -> None:
        self.dataset = dataset
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        return {
            "state": sample["state"][0].float(),
            "goal_state": sample["state"][-1].float(),
            "proprio": sample["proprio"][0].float(),
            "goal_proprio": sample["proprio"][-1].float(),
            "actions": sample["action"][: self.horizon].float(),
        }


def build_dataset(cfg: DictConfig):
    return swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)


def build_dataloaders(cfg: DictConfig):
    dataset = build_dataset(cfg)
    n_train = int(len(dataset) * cfg.train_split)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_ds = PlannerSequenceDataset(train_set, cfg.planner.horizon)
    val_ds = PlannerSequenceDataset(val_set, cfg.planner.horizon)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def make_run_dir(cfg: DictConfig) -> Path:
    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return run_dir


def q_sample(
    actions: torch.Tensor,
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(actions)
    alpha_bar = schedule.alpha_bars[timesteps].view(-1, 1, 1)
    noisy_actions = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise
    return noisy_actions, noise


@hydra.main(version_base=None, config_path="./config/train", config_name="diffusion_planner")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    model = ActionDiffusionMLP(
        action_dim=cfg.planner.action_dim,
        horizon=cfg.planner.horizon,
        state_dim=cfg.planner.state_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        time_dim=cfg.model.time_dim,
        proprio_dim=cfg.planner.proprio_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    schedule = DiffusionSchedule.cosine(cfg.diffusion.num_steps, device=device)

    best_val = float("inf")
    for epoch in range(1, cfg.trainer.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            steps = torch.randint(
                0, cfg.diffusion.num_steps, (batch["actions"].shape[0],), device=device
            )
            noisy_actions, noise = q_sample(batch["actions"], schedule, steps)
            pred_noise = model(
                noisy_actions=noisy_actions,
                timesteps=steps,
                state=batch["state"],
                goal_state=batch["goal_state"],
                proprio=batch["proprio"],
                goal_proprio=batch["goal_proprio"],
            )
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                steps = torch.randint(
                    0, cfg.diffusion.num_steps, (batch["actions"].shape[0],), device=device
                )
                noisy_actions, noise = q_sample(batch["actions"], schedule, steps)
                pred_noise = model(
                    noisy_actions=noisy_actions,
                    timesteps=steps,
                    state=batch["state"],
                    goal_state=batch["goal_state"],
                    proprio=batch["proprio"],
                    goal_proprio=batch["goal_proprio"],
                )
                val_loss += F.mse_loss(pred_noise, noise).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        latest_ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            latest_ckpt,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "val_loss": val_loss,
                },
                run_dir / f"{cfg.output_model_name}_best.ckpt",
            )


if __name__ == "__main__":
    run()
