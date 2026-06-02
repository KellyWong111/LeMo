from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box


class StateTaskCostCEMSolver:
    def __init__(
        self,
        model: Any,
        state_cost_weight: float = 0.25,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
    ) -> None:
        self.model = model
        self.state_cost_weight = state_cost_weight
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        if not hasattr(model, "state_head"):
            raise RuntimeError("StateTaskCostCEMSolver requires model.state_head")
        self.state_head = model.state_head.eval()
        self.state_head.requires_grad_(False)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}")

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

    def compute_state_cost(self, prepared: dict, model_cost: torch.Tensor) -> torch.Tensor:
        pred = prepared["predicted_emb"].detach().float()
        final_emb = pred[:, :, -1, :]
        pred_state = self.state_head(final_emb.reshape(-1, final_emb.shape[-1])).reshape(final_emb.shape[0], final_emb.shape[1], -1)
        goal_state = prepared.get("goal_state")
        if goal_state is None:
            return torch.zeros_like(model_cost)
        goal_state = goal_state.detach().float()
        if goal_state.ndim == 3:
            goal_state = goal_state.unsqueeze(1)
        if goal_state.shape[1] == 1 and goal_state.shape[1] != pred_state.shape[1]:
            goal_state = goal_state.expand(-1, pred_state.shape[1], -1, -1)
        goal_final = goal_state[:, :, -1, :].to(pred_state.device)
        return (pred_state - goal_final).square().sum(dim=-1)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "model_costs": [], "state_costs": [], "mean": [], "var": []}
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

            final_total_cost = final_model_cost = final_state_cost = None
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
                state_cost = self.compute_state_cost(prepared, model_cost)
                total_cost = model_cost + float(self.state_cost_weight) * state_cost

                topk_vals, topk_inds = torch.topk(total_cost, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                final_total_cost = topk_vals.mean(dim=1).cpu().tolist()
                final_model_cost = model_cost[batch_indices, topk_inds].mean(dim=1).cpu().tolist()
                final_state_cost = state_cost[batch_indices, topk_inds].mean(dim=1).cpu().tolist()
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_total_cost)
            outputs["model_costs"].extend(final_model_cost)
            outputs["state_costs"].extend(final_state_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(f"StateTaskCost CEM solve time: {time.time() - start_time:.4f} seconds (state_cost_weight={self.state_cost_weight})")
        return outputs
