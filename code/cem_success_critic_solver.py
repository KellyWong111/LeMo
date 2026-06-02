from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn


class SuccessCritic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SuccessCriticCEMSolver:
    def __init__(
        self,
        model: Any,
        critic_path: str,
        critic_weight: float = 1.0,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.critic_path = critic_path
        self.critic_weight = critic_weight
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        blob = torch.load(critic_path, map_location="cpu", weights_only=False)
        self.critic_mean = blob["mean"].float().to(self.device)
        self.critic_std = blob["std"].float().to(self.device)
        self.critic = SuccessCritic(int(blob["input_dim"])).to(self.device).eval()
        self.critic.load_state_dict(blob["state_dict"])
        self.critic.requires_grad_(False)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. "
                "SuccessCriticCEMSolver may not work as expected."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            mean = torch.cat([mean, torch.zeros([self.n_envs, remaining, self.action_dim])], dim=1).to(device)
        return mean, var

    def critic_features(self, prepared: dict, candidates: torch.Tensor, base_cost: torch.Tensor):
        pred = prepared["predicted_emb"].detach().float()
        goal = prepared["goal_emb"].detach().float()
        if goal.ndim == pred.ndim - 1:
            goal = goal.unsqueeze(1)
        if goal.shape[1] == 1 and pred.shape[1] != 1:
            goal = goal.expand(-1, pred.shape[1], -1, -1)
        goal_last = goal[..., -1:, :].expand_as(pred)
        dist_t = (pred - goal_last).square().sum(dim=-1).sqrt()
        final_dist = dist_t[:, :, -1]
        init_dist = (pred[:, :, 0, :] - goal_last[:, :, 0, :]).square().sum(dim=-1).sqrt()
        step_norm = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).square().sum(dim=-1).sqrt()
        topk = candidates.shape[1]
        rank = torch.arange(topk, device=pred.device, dtype=pred.dtype)[None, :].expand(candidates.shape[0], -1)
        feat = torch.stack(
            [
                base_cost.float(),
                base_cost.float(),
                rank / max(topk - 1, 1),
                final_dist,
                dist_t.mean(dim=-1),
                dist_t.min(dim=-1).values,
                dist_t.std(dim=-1),
                init_dist - final_dist,
                step_norm.mean(dim=-1),
                step_norm.std(dim=-1),
                step_norm.max(dim=-1).values,
            ],
            dim=-1,
        )
        return feat

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "model_costs": [], "critic_scores": [], "mean": [], "var": []}
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value):
                    value_batch = value_batch.unsqueeze(1)
                    value_batch = value_batch.expand(current_bs, self.num_samples, *value_batch.shape[2:])
                elif isinstance(value, np.ndarray):
                    value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value_batch

            final_total_cost = None
            final_model_cost = None
            final_critic_score = None
            for _ in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                prepared = expanded_infos.copy()
                model_cost = self.model.get_cost(prepared, candidates)
                feat = self.critic_features(prepared, candidates, model_cost)
                feat = (feat - self.critic_mean) / self.critic_std
                critic_score = self.critic(feat.reshape(-1, feat.shape[-1])).reshape_as(model_cost)
                total_cost = model_cost - float(self.critic_weight) * critic_score

                topk_vals, topk_inds = torch.topk(total_cost, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                final_total_cost = topk_vals.mean(dim=1).cpu().tolist()
                final_model_cost = model_cost[batch_indices, topk_inds].mean(dim=1).cpu().tolist()
                final_critic_score = critic_score[batch_indices, topk_inds].mean(dim=1).cpu().tolist()
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_total_cost)
            outputs["model_costs"].extend(final_model_cost)
            outputs["critic_scores"].extend(final_critic_score)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(
            f"SuccessCritic CEM solve time: {time.time() - start_time:.4f} seconds "
            f"(critic_weight={self.critic_weight})"
        )
        return outputs
