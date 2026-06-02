from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base
from topk_oracle_pilot import eval_fixed_plans, get_multistart_topk_candidates


def build_candidate_dataset(args):
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)

    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]

    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    transform = {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)}
    prepared_base = base.make_eval_like_info(raw_info, transform, process)

    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

    model = base.load_model(cfg, cache_dir=args.cache_dir)
    candidates, topk_costs = get_multistart_topk_candidates(
        model,
        prepared_base,
        action_dim=action_dim,
        horizon=int(cfg.plan_config.horizon),
        num_samples=args.num_samples,
        topk=args.topk,
        n_steps=args.cem_steps,
        seed=args.seed,
        restarts=args.restarts,
    )
    del model
    torch.cuda.empty_cache()

    plans = candidates.numpy()
    candidate_success = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        candidate_success.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(candidate_success, axis=1)
    return cfg, prepared_base, candidates.float(), torch.as_tensor(labels, dtype=torch.bool), topk_costs


def concat_prepared(items):
    keys = items[0].keys()
    out = {}
    for key in keys:
        vals = [x[key] for x in items]
        if torch.is_tensor(vals[0]):
            out[key] = torch.cat(vals, dim=0)
        else:
            out[key] = vals[0]
    return out


def build_multi_seed_candidate_dataset(args):
    seeds = [int(x) for x in str(args.train_seeds or args.seed).split(",")]
    cfg0 = None
    prepared_items = []
    candidate_items = []
    label_items = []
    cost_items = []
    original_seed = args.seed
    for seed in seeds:
        args.seed = seed
        cfg, prepared, candidates, labels, costs = build_candidate_dataset(args)
        cfg0 = cfg
        prepared_items.append(prepared)
        candidate_items.append(candidates)
        label_items.append(labels)
        cost_items.append(costs)
    args.seed = original_seed
    prepared = concat_prepared(prepared_items)
    candidates = torch.cat(candidate_items, dim=0)
    labels = torch.cat(label_items, dim=0)
    costs = torch.cat(cost_items, dim=0)
    return cfg0, prepared, candidates, labels, costs, seeds


def clone_prepared(prepared):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared.items()}


def freeze_for_cost_finetune(model, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode in ("predictor", "predictor_action"):
        for p in model.predictor.parameters():
            p.requires_grad_(True)
    if mode in ("action", "predictor_action"):
        for p in model.action_encoder.parameters():
            p.requires_grad_(True)
    if mode == "all_but_encoder":
        for name, p in model.named_parameters():
            if not name.startswith("encoder."):
                p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def compute_cost(model, prepared_base, candidates):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    return model.get_cost(prepared, candidates)


def pairwise_loss_from_cost(costs, labels, margin):
    losses = []
    stats = []
    for cost, label in zip(costs, labels):
        pos = cost[label]
        neg = cost[~label]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # We optimize model cost directly: successful candidates should have
        # lower cost than failed candidates within the same CEM candidate set.
        diff = neg[:, None] - pos[None, :]
        losses.append(torch.nn.functional.softplus(margin - diff).mean())
        stats.append((pos.mean().detach(), neg.mean().detach()))
    if not losses:
        return costs.mean() * 0.0, None
    return torch.stack(losses).mean(), stats


def save_model(model, out_dir: Path, out_name: str, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{out_name}_epoch_{epoch}_object.ckpt"
    model_cpu = model.eval().cpu()
    torch.save(model_cpu, ckpt)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-model-name", required=True)
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--l2-anchor", type=float, default=1e-4)
    parser.add_argument("--mode", choices=["predictor", "action", "predictor_action", "all_but_encoder"], default="predictor")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    cfg, prepared_base, candidates_cpu, labels_cpu, original_costs, train_seeds = build_multi_seed_candidate_dataset(args)
    model = base.load_model(cfg, cache_dir=args.cache_dir).train()
    params = freeze_for_cost_finetune(model, args.mode)
    if not params:
        raise RuntimeError(f"No trainable parameters for mode={args.mode}")

    device = torch.device("cuda")
    candidates = candidates_cpu.to(device)
    labels = labels_cpu.to(device)
    with torch.no_grad():
        anchor_costs = compute_cost(model, prepared_base, candidates).detach()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    history = []
    out_dir = Path(args.output_dir)
    out_name = args.output_model_name
    for epoch in range(1, args.epochs + 1):
        model.train()
        costs = compute_cost(model, prepared_base, candidates)
        rank_loss, stats = pairwise_loss_from_cost(costs, labels, args.margin)
        anchor = (costs - anchor_costs).pow(2).mean()
        loss = rank_loss + args.l2_anchor * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                success_cost = costs[labels].mean().item() if labels.any() else None
                fail_cost = costs[~labels].mean().item() if (~labels).any() else None
                top1_success = labels[torch.arange(labels.shape[0], device=device), costs.argmin(dim=1)].float().mean().item() * 100.0
                oracle = labels.any(dim=1).float().mean().item() * 100.0
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().item()),
                "rank_loss": float(rank_loss.detach().item()),
                "anchor": float(anchor.detach().item()),
                "success_cost": success_cost,
                "fail_cost": fail_cost,
                "cost_argmin_success": top1_success,
                "oracle": oracle,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            model.cuda()
            ckpt = save_model(model, out_dir, out_name, epoch)
            print(f"[SAVE] {ckpt}", flush=True)
            model.cuda().train()

    (out_dir / "finetune_history.json").write_text(json.dumps({
        "args": vars(args),
        "train_seeds": train_seeds,
        "labels": labels_cpu.tolist(),
        "original_topk_costs": original_costs.tolist(),
        "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
