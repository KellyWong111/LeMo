from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from action_diffusion import ActionDiffusionMLP, DiffusionSchedule

logger = logging.getLogger(__name__)


@dataclass
class DiffusionPlannerStats:
    denoise_steps: int
    num_proposals: int
    rerank_topk: int
    elapsed_sec: float


class DiffusionPlannerSolver:
    """A minimal diffusion-style planner interface for LeWM.

    This class is intentionally conservative:
    - it matches the stable_worldmodel Solver protocol used by WorldModelPolicy
    - it can run without changing the world-model cost interface
    - it leaves room for a future learned denoiser / policy checkpoint

    Current implementation:
    - samples a small set of action-sequence proposals
    - performs truncated iterative denoising around a running mean
    - optionally reranks proposals with the LeWM cost model

    This is not yet a trained DiffusionDrive-equivalent policy. It is the
    integration scaffold that lets us replace the CEM solver slot cleanly.
    """

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        num_proposals: int = 32,
        denoise_steps: int = 8,
        noise_scale: float = 1.0,
        proposal_std_decay: float = 0.75,
        rerank_topk: int = 4,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        planner_ckpt: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_proposals = num_proposals
        self.denoise_steps = denoise_steps
        self.noise_scale = noise_scale
        self.proposal_std_decay = proposal_std_decay
        self.rerank_topk = rerank_topk
        self.device = torch.device(device)
        self.planner_ckpt = planner_ckpt
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        self._configured = False
        self._planner = None
        self._schedule = None

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        print("[diffusion_planner] configure start", flush=True)
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            raise TypeError(
                f"DiffusionPlannerSolver expects a continuous Box action space, got {type(action_space)}"
            )

        # stable_worldmodel exposes a batched action space shaped like
        # (n_envs, action_dim_per_env). We only need the per-env bounds here.
        low_arr = np.asarray(action_space.low)
        high_arr = np.asarray(action_space.high)
        if low_arr.ndim > 1:
            low_arr = low_arr[0]
            high_arr = high_arr[0]
        low = low_arr.reshape(-1)
        high = high_arr.reshape(-1)
        self._base_low = torch.as_tensor(low, dtype=torch.float32, device=self.device)
        self._base_high = torch.as_tensor(high, dtype=torch.float32, device=self.device)
        print(
            f"[diffusion_planner] configure done n_envs={self.n_envs} horizon={self.horizon} action_dim={self.action_dim}",
            flush=True,
        )
        self._maybe_load_planner()

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    def init_action_plan(self, init_action: torch.Tensor | None = None) -> torch.Tensor:
        if init_action is None:
            plan = torch.zeros(
                self.n_envs, self.horizon, self.action_dim, device=self.device
            )
        else:
            plan = init_action.to(self.device)
            if plan.shape[1] < self.horizon:
                pad = torch.zeros(
                    self.n_envs,
                    self.horizon - plan.shape[1],
                    self.action_dim,
                    device=self.device,
                )
                plan = torch.cat([plan, pad], dim=1)
            else:
                plan = plan[:, : self.horizon]
        return self._clip_actions(plan)

    def _expand_action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = self._base_low.repeat(self._config.action_block)
        high = self._base_high.repeat(self._config.action_block)
        low = low.view(1, 1, self.action_dim)
        high = high.view(1, 1, self.action_dim)
        return low, high

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low, high = self._expand_action_bounds()
        return torch.clamp(actions, min=low, max=high)

    def _expand_info(self, info_dict: dict[str, Any], num_samples: int) -> dict[str, Any]:
        expanded_infos: dict[str, Any] = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                v = v.unsqueeze(1).expand(self.n_envs, num_samples, *v.shape[1:])
            elif isinstance(v, np.ndarray):
                v = np.repeat(v[:, None, ...], num_samples, axis=1)
            expanded_infos[k] = v
        return expanded_infos

    def _propose(
        self,
        current_mean: torch.Tensor,
        step_std: float,
    ) -> torch.Tensor:
        noise = torch.randn(
            self.n_envs,
            self.num_proposals,
            self.horizon,
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
        )
        proposals = current_mean.unsqueeze(1) + step_std * noise
        proposals[:, 0] = current_mean
        return self._clip_actions(proposals)

    def _maybe_load_planner(self) -> None:
        if not self.planner_ckpt:
            print("[diffusion_planner] no planner_ckpt, using proposal-only mode", flush=True)
            return
        print(f"[diffusion_planner] loading planner ckpt from {self.planner_ckpt}", flush=True)
        payload = torch.load(self.planner_ckpt, map_location="cpu")
        cfg = payload["cfg"]
        planner = ActionDiffusionMLP(
            action_dim=self.action_dim,
            horizon=self.horizon,
            state_dim=int(cfg["planner"]["state_dim"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            depth=int(cfg["model"]["depth"]),
            time_dim=int(cfg["model"]["time_dim"]),
            proprio_dim=int(cfg["planner"].get("proprio_dim", 0)),
        )
        planner.load_state_dict(payload["model"])
        planner = planner.to(self.device).eval()
        planner.requires_grad_(False)
        self._planner = planner
        self._schedule = DiffusionSchedule.cosine(int(cfg["diffusion"]["num_steps"]), device=self.device)
        print("[diffusion_planner] planner ckpt loaded", flush=True)

    def _flatten_condition(
        self,
        value: torch.Tensor | None,
        *,
        name: str,
        expected_dim: int | None = None,
        take_last: bool = True,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor, got {type(value)}")

        value = value.to(self.device).float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        elif (
            value.ndim == 2
            and expected_dim is not None
            and value.shape[0] == expected_dim
            and value.shape[1] == self.n_envs
        ):
            # Some env infos come as (D, B) instead of (B, D).
            value = value.transpose(0, 1)
        elif value.ndim >= 3:
            # WorldModelPolicy may pass history-shaped tensors, e.g. (B, T, D).
            # For planner conditioning we use one summary vector per env.
            value = value[:, -1] if take_last else value[:, 0]

        if value.ndim != 2:
            raise ValueError(f"{name} must flatten to shape (B, D), got {tuple(value.shape)}")

        if expected_dim is not None and value.shape[1] != expected_dim:
            raise ValueError(
                f"{name} has dim {value.shape[1]}, expected {expected_dim}. shape={tuple(value.shape)}"
            )
        return value

    def _planner_sample(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> torch.Tensor:
        if self._planner is None or self._schedule is None:
            raise RuntimeError("Planner checkpoint is not loaded")
        if "state" not in info_dict or "goal_state" not in info_dict:
            raise KeyError("Diffusion planner requires `state` and `goal_state` in info_dict")

        state = self._flatten_condition(info_dict["state"], name="state", expected_dim=self._planner.state_dim)
        goal_state = self._flatten_condition(
            info_dict["goal_state"], name="goal_state", expected_dim=self._planner.state_dim
        )
        proprio = self._flatten_condition(
            info_dict.get("proprio"), name="proprio", expected_dim=self._planner.proprio_dim
        )
        goal_proprio = self._flatten_condition(
            info_dict.get("goal_proprio"), name="goal_proprio", expected_dim=self._planner.proprio_dim
        )

        x = self.init_action_plan(init_action)
        print(
            f"[diffusion_planner] sample start x={tuple(x.shape)} state={tuple(state.shape)} goal_state={tuple(goal_state.shape)}",
            flush=True,
        )
        logger.info(
            "diffusion planner sample start: x=%s state=%s goal_state=%s proprio=%s goal_proprio=%s",
            tuple(x.shape),
            tuple(state.shape) if state is not None else None,
            tuple(goal_state.shape) if goal_state is not None else None,
            tuple(proprio.shape) if proprio is not None else None,
            tuple(goal_proprio.shape) if goal_proprio is not None else None,
        )
        for t in reversed(range(len(self._schedule.betas))):
            step = torch.full((self.n_envs,), t, device=self.device, dtype=torch.long)
            pred_noise = self._planner(
                noisy_actions=x,
                timesteps=step,
                state=state,
                goal_state=goal_state,
                proprio=proprio,
                goal_proprio=goal_proprio,
            )
            if t in {len(self._schedule.betas) - 1, len(self._schedule.betas) // 2, 0}:
                print(f"[diffusion_planner] denoise step {t}", flush=True)
                logger.info("diffusion planner denoise step %d/%d", t, len(self._schedule.betas) - 1)
            alpha = self._schedule.alphas[t]
            alpha_bar = self._schedule.alpha_bars[t]
            beta = self._schedule.betas[t]
            x = (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            if t > 0:
                noise = torch.randn(
                    x.shape,
                    generator=self.torch_gen,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = x + torch.sqrt(beta) * noise
            x = self._clip_actions(x)
        logger.info("diffusion planner sample done: x=%s", tuple(x.shape))
        print(f"[diffusion_planner] sample done x={tuple(x.shape)}", flush=True)
        return x

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("DiffusionPlannerSolver.configure() must be called before solve().")

        start_time = time.time()
        current_mean = self.init_action_plan(init_action)
        step_std = float(self.noise_scale)
        outputs: dict[str, Any] = {
            "costs": [],
            "mean": [],
            "stats": None,
        }

        if self._planner is not None:
            print("[diffusion_planner] solve using learned planner", flush=True)
            current_mean = self._planner_sample(info_dict, init_action)
            logger.info("diffusion planner initial proposal ready: %s", tuple(current_mean.shape))
            print(f"[diffusion_planner] initial proposal ready {tuple(current_mean.shape)}", flush=True)
            if self.rerank_topk <= 1:
                final_actions = self._clip_actions(current_mean).detach().cpu()
                elapsed = time.time() - start_time
                outputs["actions"] = final_actions
                outputs["stats"] = DiffusionPlannerStats(
                    denoise_steps=self.denoise_steps,
                    num_proposals=1,
                    rerank_topk=1,
                    elapsed_sec=elapsed,
                )
                print(f"Diffusion-style solve time: {elapsed:.4f} seconds")
                return outputs

        for _ in range(self.denoise_steps):
            print("[diffusion_planner] rerank iteration start", flush=True)
            logger.info("diffusion rerank iteration start")
            candidates = self._propose(current_mean, step_std)
            current_info = self._expand_info(info_dict, self.num_proposals)
            costs = self.model.get_cost(current_info, candidates)
            print(f"[diffusion_planner] rerank got costs {tuple(costs.shape)}", flush=True)
            logger.info("diffusion rerank iteration got costs: %s", tuple(costs.shape))

            if not isinstance(costs, torch.Tensor):
                raise TypeError(f"Expected tensor costs, got {type(costs)}")
            if costs.shape != (self.n_envs, self.num_proposals):
                raise ValueError(
                    f"Expected costs shape {(self.n_envs, self.num_proposals)}, got {tuple(costs.shape)}"
                )

            topk = min(self.rerank_topk, self.num_proposals)
            top_vals, top_inds = torch.topk(costs, k=topk, dim=1, largest=False)
            batch_idx = torch.arange(self.n_envs, device=self.device).unsqueeze(1)
            elites = candidates[batch_idx, top_inds]
            current_mean = elites.mean(dim=1)
            step_std *= self.proposal_std_decay
            outputs["costs"].append(top_vals[:, 0].detach().cpu())
            outputs["mean"].append(current_mean.detach().cpu())

        final_actions = self._clip_actions(current_mean).detach().cpu()
        elapsed = time.time() - start_time
        outputs["actions"] = final_actions
        outputs["stats"] = DiffusionPlannerStats(
            denoise_steps=self.denoise_steps,
            num_proposals=self.num_proposals,
            rerank_topk=min(self.rerank_topk, self.num_proposals),
            elapsed_sec=elapsed,
        )

        print(f"Diffusion-style solve time: {elapsed:.4f} seconds")
        return outputs
