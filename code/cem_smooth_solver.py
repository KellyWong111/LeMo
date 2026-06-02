from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box


class SmoothCEMSolver:
    """CEM with a temporal action-smoothness penalty.

    This is a behavior-constrained planning diagnostic. It keeps the stock CEM
    optimizer and elite-mean action extraction, but ranks candidates by:

        model_cost + smooth_weight * sum_t ||a_t - a_{t-1}||^2

    where the action sequence is interpreted at real environment-step
    resolution after unflattening action blocks.
    """

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        smooth_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.smooth_weight = smooth_weight
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._env_action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. "
                "SmoothCEMSolver may not work as expected."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._env_action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)
        return mean, var

    def smooth_penalty(self, candidates: torch.Tensor) -> torch.Tensor:
        if self.smooth_weight == 0:
            return torch.zeros(candidates.shape[:2], device=candidates.device)
        bsz, nsamp, horizon, _ = candidates.shape
        actions = candidates.reshape(
            bsz,
            nsamp,
            horizon * int(self._config.action_block),
            self._env_action_dim,
        )
        diffs = actions[:, :, 1:] - actions[:, :, :-1]
        return diffs.square().sum(dim=(-1, -2))

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {
            "costs": [],
            "model_costs": [],
            "smooth_penalty": [],
            "mean": [],
            "var": [],
        }

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
                    value_batch = value_batch.expand(
                        current_bs, self.num_samples, *value_batch.shape[2:]
                    )
                elif isinstance(value, np.ndarray):
                    value_batch = np.repeat(
                        value_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[key] = value_batch

            final_total_cost = None
            final_model_cost = None
            final_smooth = None

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

                model_cost = self.model.get_cost(expanded_infos.copy(), candidates)
                smooth = self.smooth_penalty(candidates)
                costs = model_cost + float(self.smooth_weight) * smooth

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = torch.arange(
                    current_bs, device=self.device
                ).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]

                gathered_model_cost = model_cost[batch_indices, topk_inds]
                gathered_smooth = smooth[batch_indices, topk_inds]
                final_total_cost = topk_vals.mean(dim=1).cpu().tolist()
                final_model_cost = gathered_model_cost.mean(dim=1).cpu().tolist()
                final_smooth = gathered_smooth.mean(dim=1).cpu().tolist()

                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_total_cost)
            outputs["model_costs"].extend(final_model_cost)
            outputs["smooth_penalty"].extend(final_smooth)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(
            f"Smooth CEM solve time: {time.time() - start_time:.4f} seconds "
            f"(smooth_weight={self.smooth_weight})"
        )
        return outputs
