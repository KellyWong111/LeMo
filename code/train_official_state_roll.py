import os
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
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import (
    LatestTrainerCheckpoint,
    ModelObjectCallBack,
    get_column_normalizer,
    get_img_preprocessor,
    resolve_resume_ckpt,
)


def build_official_jepa(cfg):
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
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
        pred_proj=predictor_proj,
    )

    state_cfg = cfg.loss.get("state_rollout", {})
    if bool(state_cfg.get("enabled", False)):
        state_dim = int(cfg.wm.state_dim)
        hidden = int(state_cfg.get("hidden_dim", 256))
        model.state_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, state_dim),
        )
    return model


def build_clean_model_for_dump(live_model, cfg):
    clean_model = build_official_jepa(cfg)
    clean_model.load_state_dict(live_model.state_dict(), strict=True)
    clean_model.eval()
    for param in clean_model.parameters():
        param.requires_grad_(False)
    return clean_model.cpu()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses, optional state rollout loss."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    state_cfg = cfg.loss.get("state_rollout", {})
    state_enabled = bool(state_cfg.get("enabled", False))
    state_rollout_loss = pred_emb.new_tensor(0.0)
    if state_enabled:
        if not hasattr(self.model, "state_head"):
            raise RuntimeError("loss.state_rollout.enabled=true requires model.state_head")
        state_target = batch["state"][:, n_preds : n_preds + pred_emb.shape[1]].to(pred_emb.device)
        state_pred = self.model.state_head(pred_emb.float()).to(state_target.dtype)
        state_rollout_loss = (state_pred - state_target).pow(2).mean()
        output["state_rollout_loss"] = state_rollout_loss
        with torch.no_grad():
            state_var = state_target.float().var(dim=(0, 1), unbiased=False).clamp_min(1e-6)
            state_dim_mse = (state_pred.float() - state_target.float()).pow(2).mean(dim=(0, 1))
            output["state_rollout_r2_mean"] = (1.0 - state_dim_mse / state_var).mean().detach()
    state_weight = float(state_cfg.get("weight", 0.0)) if state_enabled else 0.0
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"] + state_weight * state_rollout_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
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

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = build_official_jepa(cfg)

    warm_start_ckpt = cfg.get("warm_start_ckpt")
    if warm_start_ckpt:
        from utils import warm_start_model
        warm_start_model(world_model, warm_start_ckpt, strict=bool(cfg.get("warm_start_strict", False)))

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

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
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
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
    return


if __name__ == "__main__":
    run()
