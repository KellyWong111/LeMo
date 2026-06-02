from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    freq = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionDiffusionMLP(nn.Module):
    """Compact conditional denoiser for action-sequence diffusion."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        state_dim: int,
        hidden_dim: int = 512,
        depth: int = 6,
        time_dim: int = 128,
        proprio_dim: int = 0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.state_dim = state_dim
        self.proprio_dim = proprio_dim
        self.time_dim = time_dim

        traj_dim = horizon * action_dim
        cond_dim = 2 * state_dim + 2 * proprio_dim + hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(traj_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim * 4) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, traj_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        proprio: torch.Tensor | None = None,
        goal_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = noisy_actions.shape[0]
        x = noisy_actions.reshape(batch, -1)

        time_emb = sinusoidal_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        cond_parts = [state, goal_state]
        if self.proprio_dim > 0:
            if proprio is None or goal_proprio is None:
                raise ValueError("Missing proprio conditions for ActionDiffusionMLP")
            cond_parts.extend([proprio, goal_proprio])
        cond_parts.append(time_emb)
        cond = torch.cat(cond_parts, dim=1)

        h = self.input_proj(x) + self.cond_proj(cond)
        for block in self.blocks:
            h = block(h)
        pred = self.out(h)
        return pred.view(batch, self.horizon, self.action_dim)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def cosine(
        cls,
        num_steps: int,
        s: float = 0.008,
        device: str | torch.device = "cpu",
    ) -> "DiffusionSchedule":
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alpha_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(1e-5, 0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)
