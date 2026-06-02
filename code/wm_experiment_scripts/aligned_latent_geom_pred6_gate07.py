import json
from pathlib import Path
import sys

import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from hydra import compose, initialize_config_dir

sys.path.insert(0, "/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
from utils import get_column_normalizer, get_img_preprocessor


CFG_DIR = "/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean/config/train"
OUT = Path("/data1/jingyixi/wm_runs/aligned_latent_geom_pred6_gate07.json")


def build_cfg(name):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name=name)


def build_dataset(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def sample_batch(dataset, n=32, seed=42):
    g = np.random.default_rng(seed)
    idx = np.sort(g.choice(len(dataset) - 1, size=n, replace=False))
    pixels = []
    actions = []
    for i in idx:
        row = dataset[int(i)]
        pixels.append(row["pixels"])
        actions.append(row["action"])
    return {
        "pixels": torch.stack(pixels, dim=0),
        "action": torch.stack(actions, dim=0),
    }


def encode_emb(model, batch, device="cuda"):
    info = {k: v.to(device).float() for k, v in batch.items()}
    with torch.inference_mode():
        out = model.encode(info)
    return out["emb"].reshape(-1, out["emb"].shape[-1]).detach().float().cpu()


def pairwise_stats(x):
    with torch.inference_mode():
        m = min(x.shape[0], 160)
        y = x[:m]
        d = torch.cdist(y, y, p=2)
        iu = torch.triu_indices(m, m, offset=1)
        vals = d[iu[0], iu[1]]
        cov = torch.cov(x.T)
        eig = torch.linalg.eigvalsh(cov)
        eig = torch.clamp(eig, min=1e-12)
        pr = (eig.sum() ** 2 / eig.square().sum()).item()
    return {
        "pairwise_l2_mean": float(vals.mean().item()),
        "pairwise_l2_std": float(vals.std().item()),
        "cov_trace": float(eig.sum().item()),
        "eig_max": float(eig.max().item()),
        "eig_min": float(eig.min().item()),
        "participation_ratio": float(pr),
    }


def action_sensitivity(model, batch, sigma=0.15, repeats=3, device="cuda"):
    px = batch["pixels"].to(device).float()
    ac = batch["action"].to(device).float()
    with torch.inference_mode():
        b = model.encode({"pixels": px, "action": ac})
        pred0 = model.predict(b["emb"][:, -3:], b["act_emb"][:, -3:])[:, -1].detach().float().cpu()

    ratios = []
    for _ in range(repeats):
        noise = torch.randn_like(ac) * sigma
        with torch.inference_mode():
            b2 = model.encode({"pixels": px, "action": ac + noise})
            pred2 = model.predict(b2["emb"][:, -3:], b2["act_emb"][:, -3:])[:, -1].detach().float().cpu()
        dz = (pred2 - pred0).norm(dim=-1)
        da = noise[:, -3:].detach().float().cpu().reshape(noise.shape[0], -1).norm(dim=-1)
        ratios.append(dz / torch.clamp(da, min=1e-8))
    ratio = torch.cat(ratios, dim=0)
    return {
        "dz_da_ratio_mean": float(ratio.mean().item()),
        "dz_da_ratio_std": float(ratio.std().item()),
    }


def load_model(path):
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    cfg = build_cfg("lewm_encoder_moda_v14_full_visible_bs32")
    dataset = build_dataset(cfg)
    batch = sample_batch(dataset, n=32, seed=42)
    root = Path("/data1/jingyixi/.stable_worldmodel")

    models = {
        "pred6_ep4": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_4_object.ckpt",
        "gate07_ep1": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_1_object.ckpt",
        "pred6_ep7": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_7_object.ckpt",
        "gate07_ep4": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4_object.ckpt",
        "pred6_ep10": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6/lewm_encoder_moda_v14_full_visible_bs32_pred6_epoch_10_object.ckpt",
        "gate07_ep7": root / "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_7_object.ckpt",
    }

    out = {}
    for name, ckpt in models.items():
        print(f"RUN {name} {ckpt}", flush=True)
        if not ckpt.exists():
            out[name] = {"error": f"missing: {ckpt}"}
            continue
        model = load_model(str(ckpt))
        emb = encode_emb(model, batch)
        norms = emb.norm(dim=-1)
        out[name] = {
            "latent_norm_mean": float(norms.mean().item()),
            "latent_norm_std": float(norms.std().item()),
            **pairwise_stats(emb),
            **action_sensitivity(model, batch),
        }
        OUT.write_text(json.dumps(out, indent=2))
        del model
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
