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

    sens_cfg = cfg.loss.get("action_sens", {})
    sens_enabled = bool(sens_cfg.get("enabled", False))
    sens_loss = pred_emb.new_tensor(0.0)
    if sens_enabled:
        sigma = float(sens_cfg.get("sigma", 0.15))
        tau = float(sens_cfg.get("tau", 0.35))
        eps = float(sens_cfg.get("eps", 1e-6))
        noise = torch.randn_like(batch["action"][:, :ctx_len]) * sigma
        perturbed_action = batch["action"][:, :ctx_len] + noise
        perturbed_act_emb = self.model.action_encoder(perturbed_action)
        perturbed_pred = self.model.predict(ctx_emb, perturbed_act_emb)
        dz = (perturbed_pred[:, -1] - pred_emb[:, -1].detach()).norm(dim=-1)
        da = noise.reshape(noise.shape[0], -1).norm(dim=-1).clamp_min(eps)
        ratio = dz / da
        sens_loss = torch.relu(tau - ratio).mean()
        output["action_sens_loss"] = sens_loss
        output["action_sens_ratio"] = ratio.mean().detach()

    sens_weight = float(sens_cfg.get("weight", 0.0)) if sens_enabled else 0.0
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"] + sens_weight * sens_loss
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if sens_enabled:
        losses_dict[f"{stage}/action_sens_ratio"] = output["action_sens_ratio"]
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
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


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
