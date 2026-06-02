"""Training entrypoint for MoDA-style LeWM runs.

Keeps LeWM's task interface, encoder IO contract, loss, and planner entry
unchanged while replacing the latent dynamics predictor with a MoDA-style
predictor:

  - official MoDA Triton kernel
  - unified depth attention
  - RoPE
  - QK RMSNorm
  - pre-norm residual block with gated MLP

Usage:
  CUDA_VISIBLE_DEVICES=5 python train_moda.py data=pusht \
      trainer.devices=1 trainer.accelerator=gpu \
      wandb.enabled=False subdir=pusht_moda64 \
      output_model_name=lewm_moda64
"""

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
from module import Embedder, MLP, SIGReg
from moda_module import MODA_TRITON_AVAILABLE, MoDAARPredictor, MoDAVisualEncoder
from utils import (
    LatestTrainerCheckpoint,
    ModelObjectCallBack,
    get_column_normalizer,
    get_img_preprocessor,
    resolve_resume_ckpt,
)


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
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
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def ensure_moda_cfg(cfg):
    """Pin this training entrypoint to the exact MoDA setting."""
    if not MODA_TRITON_AVAILABLE:
        raise RuntimeError(
            "train_moda.py requires the official MoDA Triton kernel "
            "(fla.ops.moda.parallel_moda)."
        )
    with open_dict(cfg):
        if "use_moda" in cfg and not bool(cfg.use_moda):
            raise ValueError(
                "train_moda.py is reserved for exact MoDA runs; "
                "use train.py for the non-MoDA baseline."
            )
        cfg.use_moda = True
        if "depth_start_layer" not in cfg:
            cfg.depth_start_layer = 1
        if "encoder_moda" not in cfg:
            cfg.encoder_moda = {}
        cfg.encoder_moda.setdefault("depth", 12)
        cfg.encoder_moda.setdefault("heads", cfg.predictor.heads)
        cfg.encoder_moda.setdefault("dim_head", cfg.predictor.dim_head)
        cfg.encoder_moda.setdefault("mlp_dim", cfg.predictor.get("mlp_dim"))
        cfg.encoder_moda.setdefault("dropout", cfg.predictor.dropout)
        cfg.encoder_moda.setdefault("emb_dropout", cfg.predictor.emb_dropout)
        cfg.encoder_moda.setdefault("qk_norm", True)
        cfg.encoder_moda.setdefault("rope_theta", 10000.0)
        cfg.encoder_moda.setdefault("activation_checkpointing", True)
        cfg.encoder_moda.setdefault("initializer_range", 0.02)
        cfg.encoder_moda.setdefault("qkv_bias", False)
        cfg.encoder_moda.setdefault("norm_eps", 1e-6)
        cfg.encoder_moda.setdefault("hidden_ratio", 4)
        cfg.predictor.setdefault("activation_checkpointing", True)
        cfg.predictor.setdefault("initializer_range", 0.02)
        cfg.predictor.setdefault("qkv_bias", False)
        cfg.predictor.setdefault("norm_eps", 1e-6)
        cfg.predictor.setdefault("hidden_ratio", 4)
        if "moda_head" not in cfg:
            cfg.moda_head = {}
        cfg.moda_head.setdefault("identity_projector", True)
        cfg.moda_head.setdefault("identity_pred_proj", True)
    return cfg


def build_loader_kwargs(cfg_loader):
    """Make smoke tests robust when dataloader multiprocessing is disabled."""
    loader_kwargs = OmegaConf.to_container(cfg_loader, resolve=True)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs["prefetch_factor"] = None
        loader_kwargs["persistent_workers"] = False
    return loader_kwargs


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    cfg = ensure_moda_cfg(cfg)
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

    encoder = MoDAVisualEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.wm.embed_dim,
        use_moda=True,
        depth_start_layer=cfg.depth_start_layer,
        **cfg.encoder_moda,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    use_moda = bool(cfg.use_moda)
    depth_start_layer = int(cfg.depth_start_layer)

    predictor = MoDAARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        use_moda=use_moda,
        depth_start_layer=depth_start_layer,
        **cfg.predictor,
    )
    print(
        f"[MoDA] use_moda={use_moda}  "
        f"depth_start_layer={depth_start_layer}  "
        f"predictor_depth={cfg.predictor.depth}  "
        f"kernel=parallel_moda"
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    if cfg.moda_head.identity_projector:
        projector = torch.nn.Identity()
    else:
        projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    if cfg.moda_head.identity_pred_proj:
        predictor_proj = torch.nn.Identity()
    else:
        predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
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
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1
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
