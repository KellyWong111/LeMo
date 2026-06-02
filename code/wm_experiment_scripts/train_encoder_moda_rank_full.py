from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from moda_module import MODA_TRITON_AVAILABLE, MoDAVisualEncoder
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import (
    LatestTrainerCheckpoint,
    ModelObjectCallBack,
    get_column_normalizer,
    get_img_preprocessor,
    resolve_resume_ckpt,
    warm_start_model,
)


def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    spread_cfg = cfg.loss.get("activation_spread", {})
    spread_enabled = bool(spread_cfg.get("enabled", False))
    spread_loss = pred_emb.new_tensor(0.0)
    spread_goal_guard_loss = pred_emb.new_tensor(0.0)
    if spread_enabled:
        sigma = float(spread_cfg.get("sigma", 0.15))
        tau = float(spread_cfg.get("tau", 0.03))
        num_candidates = int(spread_cfg.get("num_candidates", 4))
        spread_mode = str(spread_cfg.get("mode", "one_step"))
        horizon = int(spread_cfg.get("horizon", 3))
        bsz = batch["action"].shape[0]
        noise = torch.randn(
            bsz,
            num_candidates,
            *batch["action"][:, :ctx_len].shape[1:],
            device=batch["action"].device,
            dtype=batch["action"].dtype,
        ) * sigma
        if spread_mode in (
            "cem_rollout",
            "cem_rollout_l2",
            "anchor_cem_rollout_l2",
            "anchor_cem_rollout_l2_goal_guard",
        ):
            total_horizon = int(spread_cfg.get("total_horizon", ctx_len + horizon))
            total_horizon = max(total_horizon, ctx_len + 1)
            future_steps = total_horizon - ctx_len
            action_dim = batch["action"].shape[-1]
            init_action = batch["action"][:, :ctx_len]
            if batch["action"].shape[1] > ctx_len:
                base_future = batch["action"][:, ctx_len : ctx_len + 1]
            else:
                base_future = init_action[:, -1:]
            base_future = base_future.expand(-1, future_steps, -1)
            future_noise = torch.randn(
                bsz,
                num_candidates,
                future_steps,
                action_dim,
                device=batch["action"].device,
                dtype=batch["action"].dtype,
            ) * sigma
            if spread_mode in ("anchor_cem_rollout_l2", "anchor_cem_rollout_l2_goal_guard"):
                future_noise[:, 0].zero_()
            candidate_future = base_future[:, None] + future_noise
            action_clip = float(spread_cfg.get("action_clip", 0.0))
            if action_clip > 0.0:
                candidate_future = torch.clamp(candidate_future, -action_clip, action_clip)
            candidate_action = torch.cat(
                [
                    init_action[:, None].expand(-1, num_candidates, -1, -1),
                    candidate_future,
                ],
                dim=2,
            )
            flat_action = candidate_action.reshape(bsz * num_candidates, total_horizon, action_dim)
            flat_emb = ctx_emb[:, None].expand(-1, num_candidates, -1, -1)
            flat_emb = flat_emb.reshape(bsz * num_candidates, ctx_len, -1).clone()
            trajectory = []
            act = flat_action[:, :ctx_len]
            future = flat_action[:, ctx_len:]
            for step in range(future_steps):
                act_emb_roll = self.model.action_encoder(act)
                next_emb = self.model.predict(flat_emb[:, -ctx_len:], act_emb_roll[:, -ctx_len:])[:, -1:]
                flat_emb = torch.cat([flat_emb, next_emb], dim=1)
                act = torch.cat([act, future[:, step : step + 1]], dim=1)
                trajectory.append(next_emb[:, 0])
            act_emb_roll = self.model.action_encoder(act)
            next_emb = self.model.predict(flat_emb[:, -ctx_len:], act_emb_roll[:, -ctx_len:])[:, -1:]
            trajectory.append(next_emb[:, 0])
            candidate_pred = torch.stack(trajectory, dim=1)
            candidate_pred = candidate_pred.reshape(bsz, num_candidates, future_steps + 1, -1)
            if spread_mode in ("anchor_cem_rollout_l2", "anchor_cem_rollout_l2_goal_guard"):
                traj = candidate_pred.reshape(bsz, num_candidates, -1).float()
                anchor = traj[:, :1]
                if bool(spread_cfg.get("detach_anchor", True)):
                    anchor = anchor.detach()
                neg = traj[:, 1:]
                anchor_dist = (neg - anchor).norm(dim=-1)
                spread = anchor_dist.mean(dim=1)
                spread_loss = torch.relu(tau - spread).mean()
                final_anchor = candidate_pred[:, :1, -1].float()
                if bool(spread_cfg.get("detach_anchor", True)):
                    final_anchor = final_anchor.detach()
                final_neg = candidate_pred[:, 1:, -1].float()
                final_spread = (final_neg - final_anchor).norm(dim=-1).mean(dim=1)
                output["activation_final_spread_mean"] = final_spread.mean().detach()
                if spread_mode == "anchor_cem_rollout_l2_goal_guard":
                    target_idx = min(candidate_pred.shape[2] - 1, tgt_emb.shape[1] - 1)
                    anchor_final = candidate_pred[:, 0, -1].float()
                    target_final = tgt_emb[:, target_idx].detach().float()
                    anchor_goal_cost = (anchor_final - target_final).pow(2).mean(dim=-1)
                    guard_target = float(spread_cfg.get("goal_guard_target", 0.0))
                    spread_goal_guard_loss = (
                        torch.relu(anchor_goal_cost - guard_target).mean()
                        if guard_target > 0.0
                        else anchor_goal_cost.mean()
                    )
                    output["activation_goal_guard_loss"] = spread_goal_guard_loss
                    output["activation_anchor_goal_cost"] = anchor_goal_cost.mean().detach()
            elif spread_mode == "cem_rollout_l2":
                traj = candidate_pred.reshape(bsz, num_candidates, -1).float()
                dists = torch.cdist(traj, traj)
                iu = torch.triu_indices(num_candidates, num_candidates, offset=1, device=dists.device)
                spread = dists[:, iu[0], iu[1]].mean(dim=1)
                spread_loss = torch.relu(tau - spread).mean()
                final_dists = torch.cdist(candidate_pred[:, :, -1].float(), candidate_pred[:, :, -1].float())
                final_spread = final_dists[:, iu[0], iu[1]].mean(dim=1)
                output["activation_final_spread_mean"] = final_spread.mean().detach()
            else:
                spread_target = candidate_pred.reshape(bsz, num_candidates, -1)
        elif spread_mode == "rollout":
            init_action = batch["action"][:, :ctx_len]
            candidate_action = init_action[:, None] + noise
            flat_action = candidate_action.reshape(bsz * num_candidates, ctx_len, -1)
            flat_emb = ctx_emb[:, None].expand(-1, num_candidates, -1, -1)
            flat_emb = flat_emb.reshape(bsz * num_candidates, ctx_len, -1).clone()
            trajectory = []
            for _ in range(horizon):
                act_emb_roll = self.model.action_encoder(flat_action)
                next_emb = self.model.predict(flat_emb[:, -ctx_len:], act_emb_roll[:, -ctx_len:])[:, -1:]
                flat_emb = torch.cat([flat_emb, next_emb], dim=1)
                flat_action = torch.cat([flat_action, flat_action[:, -1:]], dim=1)
                trajectory.append(next_emb[:, 0])
            candidate_pred = torch.stack(trajectory, dim=1)
            candidate_pred = candidate_pred.reshape(bsz, num_candidates, horizon, -1)
            spread_target = candidate_pred.reshape(bsz, num_candidates, -1)
        else:
            candidate_action = batch["action"][:, None, :ctx_len] + noise
            flat_action = candidate_action.reshape(bsz * num_candidates, ctx_len, -1)
            flat_ctx = ctx_emb[:, None].expand(-1, num_candidates, -1, -1)
            flat_ctx = flat_ctx.reshape(bsz * num_candidates, ctx_len, -1)
            candidate_act_emb = self.model.action_encoder(flat_action)
            candidate_pred = self.model.predict(flat_ctx, candidate_act_emb)[:, -1]
            spread_target = candidate_pred.reshape(bsz, num_candidates, -1)
        if spread_mode not in (
            "cem_rollout_l2",
            "anchor_cem_rollout_l2",
            "anchor_cem_rollout_l2_goal_guard",
        ):
            centered = spread_target - spread_target.mean(dim=1, keepdim=True)
            spread = centered.pow(2).mean(dim=(1, 2))
            spread_loss = torch.relu(tau - spread).mean()
        output["activation_spread_loss"] = spread_loss
        output["activation_spread_mean"] = spread.mean().detach()

    rank_cfg = cfg.loss.get("action_rank", {})
    rank_enabled = bool(rank_cfg.get("enabled", False))
    rank_loss = pred_emb.new_tensor(0.0)
    positive_guard_loss = pred_emb.new_tensor(0.0)
    if rank_enabled:
        sigma = float(rank_cfg.get("sigma", 0.15))
        margin = float(rank_cfg.get("margin", 0.02))
        guard_target = float(rank_cfg.get("positive_guard_target", 0.0))
        num_negatives = int(rank_cfg.get("num_negatives", 1))
        hard_mode = str(rank_cfg.get("hard_mode", "single"))
        positive_cost = (pred_emb[:, -1] - tgt_emb[:, -1]).pow(2).mean(dim=-1)
        if num_negatives > 1:
            bsz = batch["action"].shape[0]
            noise = torch.randn(
                bsz,
                num_negatives,
                *batch["action"][:, :ctx_len].shape[1:],
                device=batch["action"].device,
                dtype=batch["action"].dtype,
            ) * sigma
            perturbed_action = batch["action"][:, None, :ctx_len] + noise
            flat_action = perturbed_action.reshape(bsz * num_negatives, ctx_len, -1)
            flat_ctx = ctx_emb[:, None].expand(-1, num_negatives, -1, -1)
            flat_ctx = flat_ctx.reshape(bsz * num_negatives, ctx_len, -1)
            perturbed_act_emb = self.model.action_encoder(flat_action)
            perturbed_pred = self.model.predict(flat_ctx, perturbed_act_emb)
            flat_tgt = tgt_emb[:, None, -1].detach().expand(-1, num_negatives, -1)
            flat_tgt = flat_tgt.reshape(bsz * num_negatives, -1)
            neg_costs = (perturbed_pred[:, -1] - flat_tgt).pow(2)
            neg_costs = neg_costs.mean(dim=-1).reshape(bsz, num_negatives)
            if hard_mode == "closest":
                hard_idx = (neg_costs - positive_cost[:, None].detach()).abs().argmin(dim=1)
            elif hard_mode == "semi":
                above = neg_costs > positive_cost[:, None].detach()
                masked = neg_costs.masked_fill(~above, float("inf"))
                has_above = above.any(dim=1)
                semi_idx = masked.argmin(dim=1)
                fallback_idx = (neg_costs - positive_cost[:, None].detach()).abs().argmin(dim=1)
                hard_idx = torch.where(has_above, semi_idx, fallback_idx)
            elif hard_mode == "lowest":
                hard_idx = neg_costs.argmin(dim=1)
            else:
                hard_idx = neg_costs.argmax(dim=1)
            negative_cost = neg_costs.gather(1, hard_idx[:, None]).squeeze(1)
            output["action_rank_negative_cost_std"] = neg_costs.std(dim=1).mean().detach()
        else:
            noise = torch.randn_like(batch["action"][:, :ctx_len]) * sigma
            perturbed_action = batch["action"][:, :ctx_len] + noise
            perturbed_act_emb = self.model.action_encoder(perturbed_action)
            perturbed_pred = self.model.predict(ctx_emb, perturbed_act_emb)
            negative_cost = (perturbed_pred[:, -1] - tgt_emb[:, -1].detach()).pow(2).mean(dim=-1)
        rank_margin = negative_cost - positive_cost.detach()
        rank_loss = torch.relu(margin - rank_margin).mean()
        if guard_target > 0.0:
            positive_guard_loss = torch.relu(positive_cost - guard_target).mean()
        output["action_rank_loss"] = rank_loss
        output["action_positive_guard_loss"] = positive_guard_loss
        output["action_rank_margin"] = rank_margin.mean().detach()
        output["action_rank_positive_cost"] = positive_cost.mean().detach()
        output["action_rank_negative_cost"] = negative_cost.mean().detach()

    subspace_cfg = cfg.loss.get("action_subspace", {})
    subspace_enabled = bool(subspace_cfg.get("enabled", False))
    subspace_loss = pred_emb.new_tensor(0.0)
    if subspace_enabled:
        if not hasattr(self.model, "action_subspace_head"):
            raise RuntimeError("loss.action_subspace.enabled=true requires model.action_subspace_head")
        sigma = float(subspace_cfg.get("sigma", 0.15))
        tau = float(subspace_cfg.get("tau", 1.0))
        num_candidates = int(subspace_cfg.get("num_candidates", 4))
        total_horizon = int(subspace_cfg.get("total_horizon", ctx_len + 3))
        total_horizon = max(total_horizon, ctx_len + 1)
        future_steps = total_horizon - ctx_len
        bsz = batch["action"].shape[0]
        action_dim = batch["action"].shape[-1]
        init_action = batch["action"][:, :ctx_len]
        if batch["action"].shape[1] > ctx_len:
            base_future = batch["action"][:, ctx_len : ctx_len + 1]
        else:
            base_future = init_action[:, -1:]
        base_future = base_future.expand(-1, future_steps, -1)
        future_noise = torch.randn(
            bsz,
            num_candidates,
            future_steps,
            action_dim,
            device=batch["action"].device,
            dtype=batch["action"].dtype,
        ) * sigma
        future_noise[:, 0].zero_()
        candidate_future = base_future[:, None] + future_noise
        action_clip = float(subspace_cfg.get("action_clip", 0.0))
        if action_clip > 0.0:
            candidate_future = torch.clamp(candidate_future, -action_clip, action_clip)
        candidate_action = torch.cat(
            [
                init_action[:, None].expand(-1, num_candidates, -1, -1),
                candidate_future,
            ],
            dim=2,
        )
        flat_action = candidate_action.reshape(bsz * num_candidates, total_horizon, action_dim)
        flat_emb = ctx_emb[:, None].expand(-1, num_candidates, -1, -1)
        flat_emb = flat_emb.reshape(bsz * num_candidates, ctx_len, -1).clone()
        act = flat_action[:, :ctx_len]
        future = flat_action[:, ctx_len:]
        trajectory = []
        for step in range(future_steps):
            act_emb_roll = self.model.action_encoder(act)
            next_emb = self.model.predict(flat_emb[:, -ctx_len:], act_emb_roll[:, -ctx_len:])[:, -1:]
            flat_emb = torch.cat([flat_emb, next_emb], dim=1)
            act = torch.cat([act, future[:, step : step + 1]], dim=1)
            trajectory.append(next_emb[:, 0])
        subspace_input = torch.stack(trajectory, dim=1).float()
        if bool(subspace_cfg.get("detach_base", False)):
            subspace_input = subspace_input.detach()
        projected = self.model.action_subspace_head(subspace_input)
        if bool(subspace_cfg.get("normalize", False)):
            projected = torch.nn.functional.normalize(projected, dim=-1)
        projected = projected.reshape(bsz, num_candidates, future_steps, -1)
        rollout_latent = subspace_input.reshape(bsz, num_candidates, future_steps, -1)
        proj_traj = projected.reshape(bsz, num_candidates, -1)
        anchor = proj_traj[:, :1]
        if bool(subspace_cfg.get("detach_anchor", True)):
            anchor = anchor.detach()
        subspace_mode = str(subspace_cfg.get("mode", "spread"))
        if subspace_mode == "near_tie":
            target_idx = min(rollout_latent.shape[2] - 1, tgt_emb.shape[1] - 1)
            target_final = tgt_emb[:, target_idx].detach().float()
            goal_costs = (rollout_latent[:, :, -1].float() - target_final[:, None]).pow(2).mean(dim=-1)
            anchor_cost = goal_costs[:, :1].detach()
            neg_costs = goal_costs[:, 1:]
            above = neg_costs > anchor_cost
            tie_window = float(subspace_cfg.get("tie_window", 0.0))
            if tie_window > 0.0:
                above = above & ((neg_costs - anchor_cost) <= tie_window)
            masked = neg_costs.masked_fill(~above, float("inf"))
            has_near_tie = torch.isfinite(masked).any(dim=1)
            near_idx = masked.argmin(dim=1)
            fallback_idx = (neg_costs - anchor_cost).abs().argmin(dim=1)
            chosen_idx = torch.where(has_near_tie, near_idx, fallback_idx) + 1
            chosen = proj_traj.gather(
                1,
                chosen_idx[:, None, None].expand(-1, 1, proj_traj.shape[-1]),
            )
            subspace_spread = (chosen[:, 0] - anchor[:, 0]).norm(dim=-1)
            output["action_subspace_near_tie_rate"] = has_near_tie.float().mean().detach()
            chosen_goal_margin = goal_costs.gather(1, chosen_idx[:, None]).squeeze(1) - goal_costs[:, 0]
            output["action_subspace_chosen_goal_margin"] = chosen_goal_margin.mean().detach()
        else:
            neg = proj_traj[:, 1:]
            subspace_spread = (neg - anchor).norm(dim=-1).mean(dim=1)
        subspace_loss = torch.relu(tau - subspace_spread).mean()
        output["action_subspace_loss"] = subspace_loss
        output["action_subspace_spread"] = subspace_spread.mean().detach()

    rank_weight = float(rank_cfg.get("weight", 0.0)) if rank_enabled else 0.0
    guard_weight = float(rank_cfg.get("positive_guard_weight", 0.0)) if rank_enabled else 0.0
    spread_weight = float(spread_cfg.get("weight", 0.0)) if spread_enabled else 0.0
    spread_goal_guard_weight = (
        float(spread_cfg.get("goal_guard_weight", 0.0)) if spread_enabled else 0.0
    )
    subspace_weight = float(subspace_cfg.get("weight", 0.0)) if subspace_enabled else 0.0
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + spread_weight * spread_loss
        + spread_goal_guard_weight * spread_goal_guard_loss
        + rank_weight * rank_loss
        + guard_weight * positive_guard_loss
        + subspace_weight * subspace_loss
    )
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if spread_enabled:
        losses_dict[f"{stage}/activation_spread_mean"] = output["activation_spread_mean"]
        if "activation_final_spread_mean" in output:
            losses_dict[f"{stage}/activation_final_spread_mean"] = output["activation_final_spread_mean"]
        if "activation_anchor_goal_cost" in output:
            losses_dict[f"{stage}/activation_anchor_goal_cost"] = output["activation_anchor_goal_cost"]
    if rank_enabled:
        losses_dict[f"{stage}/action_rank_margin"] = output["action_rank_margin"]
        losses_dict[f"{stage}/action_rank_positive_cost"] = output["action_rank_positive_cost"]
        losses_dict[f"{stage}/action_rank_negative_cost"] = output["action_rank_negative_cost"]
    if subspace_enabled:
        losses_dict[f"{stage}/action_subspace_spread"] = output["action_subspace_spread"]
        if "action_subspace_near_tie_rate" in output:
            losses_dict[f"{stage}/action_subspace_near_tie_rate"] = output["action_subspace_near_tie_rate"]
        if "action_subspace_chosen_goal_margin" in output:
            losses_dict[f"{stage}/action_subspace_chosen_goal_margin"] = output["action_subspace_chosen_goal_margin"]
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def ensure_encoder_moda_cfg(cfg):
    if not MODA_TRITON_AVAILABLE:
        raise RuntimeError(
            "train_encoder_moda.py requires the official MoDA Triton kernel "
            "(fla.ops.moda.parallel_moda)."
        )
    with open_dict(cfg):
        cfg.encoder_moda.setdefault("chunk_visible", False)
    return cfg


def build_loader_kwargs(cfg_loader):
    loader_kwargs = OmegaConf.to_container(cfg_loader, resolve=True)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs["prefetch_factor"] = None
        loader_kwargs["persistent_workers"] = False
    return loader_kwargs


def build_clean_encoder_moda_jepa(cfg):
    encoder = MoDAVisualEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.wm.embed_dim,
        **cfg.encoder_moda,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    subspace_cfg = cfg.loss.get("action_subspace", {})
    if bool(subspace_cfg.get("enabled", False)):
        out_dim = int(subspace_cfg.get("dim", 64))
        model.action_subspace_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, out_dim, bias=False),
        )
    return model


def build_clean_model_for_dump(live_model, cfg):
    clean_model = build_clean_encoder_moda_jepa(cfg)
    clean_model.load_state_dict(live_model.state_dict(), strict=True)
    clean_model.eval()
    for param in clean_model.parameters():
        param.requires_grad_(False)
    return clean_model.cpu()


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    cfg = ensure_encoder_moda_cfg(cfg)
    loader_kwargs = build_loader_kwargs(cfg.loader)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )

    world_model = build_clean_encoder_moda_jepa(cfg)
    warm_start_ckpt = cfg.get("warm_start_ckpt")
    if warm_start_ckpt:
        warm_start_model(
            world_model,
            warm_start_ckpt,
            strict=bool(cfg.get("warm_start_strict", False)),
        )
    hidden_dim = world_model.encoder.config.hidden_size

    print(
        "[EncoderMoDA] "
        f"encoder_depth={int(cfg.encoder_moda.depth)} "
        f"encoder_heads={int(cfg.encoder_moda.heads)} "
        f"encoder_depth_start_layer={int(cfg.encoder_moda.depth_start_layer)} "
        f"chunk_visible={bool(cfg.encoder_moda.chunk_visible)} "
        f"kernel={'parallel_moda_chunk_visible' if bool(cfg.encoder_moda.chunk_visible) else 'parallel_moda_v14_visible'} "
        f"predictor_depth={int(cfg.predictor.depth)}"
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
        model_builder=partial(build_clean_model_for_dump, cfg=cfg),
    )
    weights_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    latest_checkpoint_callback = LatestTrainerCheckpoint(weights_ckpt_path)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, latest_checkpoint_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=resolve_resume_ckpt(run_dir, cfg.output_model_name),
    )

    manager()


if __name__ == "__main__":
    run()
