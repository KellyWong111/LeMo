from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

class BestCandidateCEMSolver:
    """CEM variant that executes the best sampled candidate instead of elite mean.

    The stock stable_worldmodel CEM solver updates its distribution with top-k
    candidates and returns the final elite mean as the action sequence. This
    variant keeps the same optimization loop but returns the lowest-cost sampled
    candidate from the final CEM iteration. It is intended as a diagnostic for
    whether planning failures come from action selection/averaging rather than
    candidate generation.
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
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. "
                "BestCandidateCEMSolver may not work as expected."
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

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": [], "best_costs": []}

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        best_actions = torch.zeros_like(mean)

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            batch_best_actions = batch_mean.clone()
            final_batch_cost = None
            final_best_cost = None

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

                costs = self.model.get_cost(expanded_infos.copy(), candidates)
                assert isinstance(costs, torch.Tensor)
                assert costs.shape == (current_bs, self.num_samples)

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = torch.arange(
                    current_bs, device=self.device
                ).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]

                best_inds = topk_inds[:, 0]
                batch_best_actions = candidates[
                    torch.arange(current_bs, device=self.device), best_inds
                ]
                final_best_cost = topk_vals[:, 0].cpu().tolist()
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            best_actions[start_idx:end_idx] = batch_best_actions
            outputs["costs"].extend(final_batch_cost)
            outputs["best_costs"].extend(final_best_cost)

        outputs["actions"] = best_actions.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(f"BestCandidate CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs
