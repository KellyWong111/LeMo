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
from torch import nn

import analyze_cem_margin as base
from topk_oracle_pilot import eval_fixed_plans, get_multistart_topk_candidates


class SuccessCritic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def clone_prepared(prepared):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared.items()}


@torch.inference_mode()
def extract_scalar_features(model, prepared_base, candidates: torch.Tensor, costs: torch.Tensor):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    cand = candidates.to("cuda")
    model_cost = model.get_cost(prepared, cand).detach().float()
    pred = prepared["predicted_emb"].detach().float()
    goal = prepared["goal_emb"].detach().float()
    if goal.ndim == pred.ndim - 1:
        goal = goal.unsqueeze(1)
    if goal.shape[1] == 1 and pred.shape[1] != 1:
        goal = goal.expand(-1, pred.shape[1], -1, -1)
    goal_last = goal[..., -1:, :].expand_as(pred)
    dist_t = (pred - goal_last).square().sum(dim=-1).sqrt()
    final_dist = dist_t[:, :, -1]
    init_dist = (pred[:, :, 0, :] - goal_last[:, :, 0, :]).square().sum(dim=-1).sqrt()
    step_norm = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).square().sum(dim=-1).sqrt()
    rank = torch.arange(topk, device=pred.device, dtype=pred.dtype)[None, :].expand(num_envs, -1)
    feat = torch.stack(
        [
            costs.to(pred.device).float(),
            model_cost,
            rank / max(topk - 1, 1),
            final_dist,
            dist_t.mean(dim=-1),
            dist_t.min(dim=-1).values,
            dist_t.std(dim=-1),
            init_dist - final_dist,
            step_norm.mean(dim=-1),
            step_norm.std(dim=-1),
            step_norm.max(dim=-1).values,
        ],
        dim=-1,
    )
    return feat.detach().cpu()


def build_data_for_seed(args, seed: int):
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
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
        seed=seed,
        restarts=args.restarts,
    )
    labels = []
    plans = candidates.numpy()
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = torch.as_tensor(np.stack(labels, axis=1), dtype=torch.float32)
    feats = extract_scalar_features(model, prepared_base, candidates.float(), topk_costs.float())
    del model
    torch.cuda.empty_cache()
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-seeds", default="42,43,44")
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    feats_all = []
    labels_all = []
    for seed in [int(x) for x in args.train_seeds.split(",")]:
        feats, labels = build_data_for_seed(args, seed)
        feats_all.append(feats)
        labels_all.append(labels)
    x = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats_all], dim=0).float()
    y = torch.cat([l.reshape(-1) for l in labels_all], dim=0).float()
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    device = torch.device("cuda")
    model = SuccessCritic(x.shape[-1]).to(device)
    x = x.to(device)
    y = y.to(device)
    pos_weight = ((y == 0).sum() / (y == 1).sum().clamp_min(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(1, args.epochs + 1):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            pred = (logits.detach().sigmoid() >= 0.5).float()
            acc = (pred == y).float().mean().item()
            auc_proxy = logits[y == 1].mean().item() - logits[y == 0].mean().item() if (y == 1).any() and (y == 0).any() else 0.0
            print(json.dumps({"epoch": epoch, "loss": loss.item(), "acc": acc, "pos_minus_neg_logit": auc_proxy}), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "mean": mean,
            "std": std,
            "input_dim": int(mean.numel()),
            "args": vars(args),
        },
        out,
    )
    print(f"[SAVE] {out}", flush=True)


if __name__ == "__main__":
    main()
