import argparse
import json
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from jepa import JEPA
from module import modulate
from utils import get_column_normalizer, get_img_preprocessor


def build_loader_kwargs(cfg_loader):
    loader_kwargs = OmegaConf.to_container(cfg_loader, resolve=True)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs["prefetch_factor"] = None
        loader_kwargs["persistent_workers"] = False
    return loader_kwargs


def build_dataset(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def tensor_stats(x):
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "mean_abs": float(x.abs().mean().item()),
        "max_abs": float(x.abs().max().item()),
        "rms": float(x.square().mean().sqrt().item()),
    }


def weight_stats(w):
    w = w.detach().float()
    return {
        "l2": float(w.norm().item()),
        "rms": float(w.square().mean().sqrt().item()),
        "mean_abs": float(w.abs().mean().item()),
        "max_abs": float(w.abs().max().item()),
    }


def get_last_linear(seq):
    if isinstance(seq, torch.nn.Sequential):
        return seq[-1]
    raise TypeError(f"Expected nn.Sequential, got {type(seq)}")


def extract_param_stats(model):
    predictor = model.predictor
    layers = predictor.transformer.layers
    out = []
    for layer_idx, block in enumerate(layers):
        layer_stats = {"layer": layer_idx}
        if hasattr(block, "adaLN_modulation"):
            mod = get_last_linear(block.adaLN_modulation)
            weight_chunks = mod.weight.detach().float().chunk(6, dim=0)
            bias_chunks = mod.bias.detach().float().chunk(6, dim=0)
            names = [
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            ]
            layer_stats["adaln"] = {}
            for name, w_chunk, b_chunk in zip(names, weight_chunks, bias_chunks):
                layer_stats["adaln"][name] = {
                    "weight": weight_stats(w_chunk),
                    "bias": tensor_stats(b_chunk),
                }
        if hasattr(block, "attn"):
            attn = block.attn
            layer_stats["attn"] = {}
            if hasattr(attn, "to_qkv"):
                q, k, v = attn.to_qkv.weight.detach().float().chunk(3, dim=0)
                layer_stats["attn"]["q_proj"] = weight_stats(q)
                layer_stats["attn"]["k_proj"] = weight_stats(k)
                layer_stats["attn"]["v_proj"] = weight_stats(v)
            if hasattr(attn, "to_out"):
                out_layer = attn.to_out[0] if isinstance(attn.to_out, torch.nn.Sequential) else attn.to_out
                if hasattr(out_layer, "weight"):
                    layer_stats["attn"]["out_proj"] = weight_stats(out_layer.weight)
        out.append(layer_stats)
    return out


def compute_qkv(attn_module, attn_input):
    normed = attn_module.norm(attn_input)
    qkv = attn_module.to_qkv(normed).chunk(3, dim=-1)
    heads = attn_module.heads
    q, k, v = (rearrange(t, "b t (h d) -> b t h d", h=heads) for t in qkv)
    return q, k, v


def causal_seq_probs(q, k, scale):
    q_hf = rearrange(q, "b t h d -> b h t d")
    k_hf = rearrange(k, "b t h d -> b h t d")
    scores = torch.einsum("bhtd,bhsd->bhts", q_hf, k_hf) * scale
    t = scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(t, t, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return torch.softmax(scores, dim=-1)


def depth_probs(q, cached_k, scale, t):
    l = cached_k.shape[1] // t
    kd = cached_k.view(cached_k.shape[0], t, l, cached_k.shape[2], cached_k.shape[3])
    q_hf = rearrange(q, "b t h d -> b h t d")
    kd_hf = rearrange(kd, "b t l h d -> b h t l d")
    scores = torch.einsum("bhtd,bhtld->bhtl", q_hf, kd_hf) * scale
    return torch.softmax(scores, dim=-1)


def unified_budget_probs(q, k, cached_k, scale):
    t = q.shape[1]
    seq_probs = causal_seq_probs(q, k, scale)
    l = cached_k.shape[1] // t
    kd = cached_k.view(cached_k.shape[0], t, l, cached_k.shape[2], cached_k.shape[3])
    q_hf = rearrange(q, "b t h d -> b h t d")
    k_hf = rearrange(k, "b t h d -> b h t d")
    kd_hf = rearrange(kd, "b t l h d -> b h t l d")
    seq_scores = torch.einsum("bhtd,bhsd->bhts", q_hf, k_hf) * scale
    depth_scores = torch.einsum("bhtd,bhtld->bhtl", q_hf, kd_hf) * scale
    causal_mask = torch.triu(
        torch.ones(t, t, device=seq_scores.device, dtype=torch.bool),
        diagonal=1,
    )
    seq_scores = seq_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(torch.cat([seq_scores, depth_scores], dim=-1), dim=-1)
    return probs, seq_probs


def analyze_predictor_batch(predictor, x, c):
    predictor.eval()
    x = x + predictor.pos_embedding[:, : x.size(1)]
    x = predictor.dropout(x)
    transformer = predictor.transformer
    if hasattr(transformer, "input_proj"):
        x = transformer.input_proj(x)
    if c is not None and hasattr(transformer, "cond_proj"):
        c = transformer.cond_proj(c)

    t = x.shape[1]
    k_cache_list = []
    v_cache_list = []
    cache_layer_ids = []
    layer_results = []

    for layer_idx, block in enumerate(transformer.layers):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.adaLN_modulation(c).chunk(6, dim=-1)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q, k, v = compute_qkv(block.attn, attn_input)

        has_depth = False
        cached_k = None
        cached_v = None
        source_layers = []
        if transformer.__class__.__name__.startswith("MoDA"):
            if (
                getattr(transformer, "use_moda", True)
                and layer_idx >= getattr(transformer, "depth_start_layer", 1)
                and k_cache_list
            ):
                cached_k, cached_v = transformer._build_depth_cache(k_cache_list, v_cache_list, t)
                source_layers = cache_layer_ids[-len(k_cache_list if getattr(transformer, "cache_window", 0) == 0 else k_cache_list[-int(transformer.cache_window):]):]
                has_depth = cached_k is not None
        elif transformer.__class__.__name__.startswith("FDA"):
            if k_cache_list:
                cached_k, cached_v = transformer._build_depth_cache(k_cache_list, v_cache_list, t)
                source_layers = cache_layer_ids.copy()
                has_depth = cached_k is not None

        layer_info = {
            "layer": layer_idx,
            "gate_msa": tensor_stats(gate_msa),
            "gate_mlp": tensor_stats(gate_mlp),
            "k_norm": float(k.detach().float().norm(dim=-1).mean().item()),
            "v_norm": float(v.detach().float().norm(dim=-1).mean().item()),
            "has_depth": bool(has_depth),
            "source_layers": source_layers,
        }

        attn_kind = block.attn.__class__.__name__
        if "MoDA" in attn_kind:
            if has_depth:
                probs, _ = unified_budget_probs(q, k, cached_k, block.attn.scale)
                seq_mass = probs[..., :t].sum(dim=-1).mean().item()
                depth_mass = probs[..., t:].sum(dim=-1).mean().item()
                depth_only = probs[..., t:].mean(dim=(0, 1, 2))
                layer_info["seq_mass"] = float(seq_mass)
                layer_info["depth_mass"] = float(depth_mass)
                layer_info["depth_source_mass"] = {
                    str(src): float(depth_only[i].item())
                    for i, src in enumerate(source_layers)
                }
            else:
                layer_info["seq_mass"] = 1.0
                layer_info["depth_mass"] = 0.0
                layer_info["depth_source_mass"] = {}
            attn_out, cur_k, cur_v = block.attn(attn_input, cached_k, cached_v)
        elif "FDA" in attn_kind:
            if has_depth:
                probs = depth_probs(q, cached_k, block.attn.scale, t)
                depth_only = probs.mean(dim=(0, 1, 2))
                layer_info["seq_mass"] = 0.0
                layer_info["depth_mass"] = 1.0
                layer_info["depth_source_mass"] = {
                    str(src): float(depth_only[i].item())
                    for i, src in enumerate(source_layers)
                }
            else:
                layer_info["seq_mass"] = 0.0
                layer_info["depth_mass"] = 0.0
                layer_info["depth_source_mass"] = {}
            attn_out, cur_k, cur_v = block.attn(attn_input, cached_k, cached_v)
        else:
            probs = causal_seq_probs(q, k, block.attn.scale)
            layer_info["seq_mass"] = 1.0
            layer_info["depth_mass"] = 0.0
            layer_info["depth_source_mass"] = {}
            attn_out = block.attn(attn_input)
            cur_k, cur_v = k, v

        x = x + gate_msa * attn_out
        x = x + gate_mlp * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))

        if hasattr(transformer, "detach_cache") and getattr(transformer, "detach_cache", False):
            cur_k = cur_k.detach()
            cur_v = cur_v.detach()
        k_cache_list.append(cur_k)
        v_cache_list.append(cur_v)
        cache_layer_ids.append(layer_idx)
        layer_results.append(layer_info)

    return layer_results


def merge_layer_stats(layer_acc, batch_layers):
    if not layer_acc:
        return batch_layers
    for acc, cur in zip(layer_acc, batch_layers):
        for key in ["k_norm", "v_norm", "seq_mass", "depth_mass"]:
            acc[key] += cur[key]
        for gate_key in ["gate_msa", "gate_mlp"]:
            for stat_name in acc[gate_key]:
                acc[gate_key][stat_name] += cur[gate_key][stat_name]
        cur_sources = cur.get("depth_source_mass", {})
        acc_sources = acc.setdefault("depth_source_mass", {})
        for src, val in cur_sources.items():
            acc_sources[src] = acc_sources.get(src, 0.0) + val
    return layer_acc


def finalize_layer_stats(layer_stats, num_batches):
    for layer in layer_stats:
        for key in ["k_norm", "v_norm", "seq_mass", "depth_mass"]:
            layer[key] /= num_batches
        for gate_key in ["gate_msa", "gate_mlp"]:
            for stat_name in layer[gate_key]:
                layer[gate_key][stat_name] /= num_batches
        if "depth_source_mass" in layer:
            for src in list(layer["depth_source_mass"].keys()):
                layer["depth_source_mass"][src] /= num_batches
    return layer_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    dataset = build_dataset(cfg)
    loader_cfg = build_loader_kwargs(cfg.loader)
    loader_cfg["batch_size"] = args.batch_size
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    rnd_gen = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, generator=rnd_gen, **loader_cfg)

    activation_stats = None
    history_size = int(cfg.wm.history_size)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            batch_layers = analyze_predictor_batch(model.predictor, ctx_emb, ctx_act)
            activation_stats = merge_layer_stats(activation_stats, batch_layers)

    activation_stats = finalize_layer_stats(activation_stats, max(1, args.num_batches))
    result = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "param_stats": extract_param_stats(model),
        "activation_stats": activation_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(
        {
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "layers": len(result["activation_stats"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
