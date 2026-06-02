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


class StateHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def clone_prepared(prepared):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared.items()}


def pick_indices(valid_indices, count: int, seed: int):
    rng = np.random.default_rng(seed)
    count = min(count, len(valid_indices))
    picked = rng.choice(len(valid_indices), size=count, replace=False)
    return np.sort(np.asarray(valid_indices)[picked])


def prepare_rows(cfg, dataset, process, indices):
    rows = dataset.get_row_data(indices)
    raw = {
        "pixels": rows["pixels"][:, None, ...],
        "state": rows["state"][:, None, ...],
    }
    return base.make_eval_like_info(raw, {"pixels": base.img_transform(cfg)}, process)


@torch.inference_mode()
def encode_states(model, cfg, dataset, process, indices, batch_size: int):
    embs = []
    targets = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        prepared = prepare_rows(cfg, dataset, process, batch_idx)
        info = {"pixels": prepared["pixels"].to("cuda")}
        encoded = model.encode(info)
        embs.append(encoded["emb"][:, -1].detach().cpu())
        targets.append(prepared["state"][:, -1].float().cpu())
    return torch.cat(embs, dim=0), torch.cat(targets, dim=0)


def train_state_head(model, cfg, dataset, process, valid_indices, args):
    train_idx = pick_indices(valid_indices, args.train_samples, args.seed + 17)
    val_idx = pick_indices(valid_indices, args.val_samples, args.seed + 191)
    x_train, y_train = encode_states(model, cfg, dataset, process, train_idx, args.encode_batch)
    x_val, y_val = encode_states(model, cfg, dataset, process, val_idx, args.encode_batch)

    head = StateHead(x_train.shape[-1], y_train.shape[-1]).to("cuda")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    x_train = x_train.to("cuda")
    y_train = y_train.to("cuda")
    x_val_cuda = x_val.to("cuda")
    y_val_cuda = y_val.to("cuda")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(x_train.shape[0], generator=gen, device="cuda")
        total = 0.0
        for start in range(0, x_train.shape[0], args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = head(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_pred = head(x_val_cuda)
                val_mse = loss_fn(val_pred, y_val_cuda).item()
                target_var = y_val_cuda.var(dim=0, unbiased=False).clamp_min(1e-6)
                dim_mse = (val_pred - y_val_cuda).square().mean(dim=0)
                r2 = 1.0 - dim_mse / target_var
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_mse": total / x_train.shape[0],
                        "val_mse": val_mse,
                        "val_r2_mean": float(r2.mean().item()),
                        "val_r2_min": float(r2.min().item()),
                    }
                ),
                flush=True,
            )
    return head.eval(), {
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "state_dim": int(y_train.shape[-1]),
        "emb_dim": int(x_train.shape[-1]),
    }


@torch.inference_mode()
def task_state_cost(model, head, prepared_base, candidates):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    cand = candidates.to("cuda")
    latent_cost = model.get_cost(prepared, cand).detach().float().cpu()
    pred = prepared["predicted_emb"].detach().float()
    final_emb = pred[:, :, -1, :]
    pred_state = head(final_emb.reshape(-1, final_emb.shape[-1])).reshape(num_envs, topk, -1)
    goal_state = prepared["goal_state"].detach().float()
    if goal_state.ndim == 3:
        goal_state = goal_state.unsqueeze(1)
    if goal_state.shape[1] == 1 and topk != 1:
        goal_state = goal_state.expand(-1, topk, -1, -1)
    goal_final = goal_state[:, :, -1, :]
    cost = (pred_state - goal_final).square().sum(dim=-1)
    return latent_cost, cost.detach().cpu(), pred_state.detach().cpu(), goal_final.detach().cpu()


def success_rate_for_argmin(labels: np.ndarray, costs: np.ndarray):
    pick = np.argmin(costs, axis=1)
    return float(labels[np.arange(labels.shape[0]), pick].mean() * 100.0), pick.tolist()


def rowwise_pairwise_auc(labels: np.ndarray, costs: np.ndarray):
    aucs = []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) == 0 or len(neg) == 0:
            continue
        aucs.append(float((pos[:, None] < neg[None, :]).mean()))
    return None if not aucs else float(np.mean(aucs))


def cost_gap(labels: np.ndarray, costs: np.ndarray):
    gaps = []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) == 0 or len(neg) == 0:
            continue
        gaps.append(float(neg.mean() - pos.mean()))
    return None if not gaps else float(np.mean(gaps))


def zscore_rows(x: np.ndarray):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def evaluate_seed(model, head, cfg, dataset, process, valid_indices, seed: int, args):
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]
    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared_base = base.make_eval_like_info(
        raw_info,
        {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)},
        process,
    )

    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

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
    latent_cost, state_cost, _, _ = task_state_cost(model, head, prepared_base, candidates.float())
    plans = candidates.numpy()
    labels = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(labels, axis=1)
    latent_np = latent_cost.numpy()
    state_np = state_cost.numpy()

    latent_top1_rate = float(labels[:, 0].mean() * 100.0)
    state_top1_rate, state_pick = success_rate_for_argmin(labels, state_np)
    oracle_rate = float(labels.any(axis=1).mean() * 100.0)

    mixed = {}
    latent_z = zscore_rows(latent_np)
    state_z = zscore_rows(state_np)
    for beta in args.beta_grid:
        mixed_cost = latent_z + beta * state_z
        rate, picks = success_rate_for_argmin(labels, mixed_cost)
        mixed[str(beta)] = {"success_rate": rate, "picks": picks}

    return {
        "seed": seed,
        "indices": indices.tolist(),
        "latent_top1_success_rate": latent_top1_rate,
        "state_top1_success_rate": state_top1_rate,
        "oracle_topk_success_rate": oracle_rate,
        "latent_pairwise_auc": rowwise_pairwise_auc(labels, latent_np),
        "state_pairwise_auc": rowwise_pairwise_auc(labels, state_np),
        "latent_success_gap": cost_gap(labels, latent_np),
        "state_success_gap": cost_gap(labels, state_np),
        "state_picks": state_pick,
        "mixed": mixed,
        "candidate_successes_by_rank": [labels[:, rank].tolist() for rank in range(args.topk)],
        "latent_costs": latent_np.tolist(),
        "state_costs": state_np.tolist(),
        "original_topk_costs": topk_costs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-seeds", default="42,43,44")
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--beta-grid", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--use-model-state-head", action="store_true")
    args = parser.parse_args()
    args.beta_grid = [float(x) for x in args.beta_grid.split(",") if x]

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = args.policy
    cfg.eval.num_eval = args.num_eval
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)

    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    model = base.load_model(cfg, cache_dir=args.cache_dir)
    if args.use_model_state_head:
        if not hasattr(model, "state_head"):
            raise RuntimeError("--use-model-state-head requires checkpoint with model.state_head")
        head = model.state_head.eval()
        train_summary = {"source": "model.state_head"}
    else:
        head, train_summary = train_state_head(model, cfg, dataset, process, valid_indices, args)

    seed_results = []
    for seed in [int(x) for x in args.eval_seeds.split(",") if x]:
        print(f"[EVAL_SEED] {seed}", flush=True)
        seed_results.append(evaluate_seed(model, head, cfg, dataset, process, valid_indices, seed, args))

    summary = {
        "policy": args.policy,
        "settings": {k: v for k, v in vars(args).items() if k != "beta_grid"} | {"beta_grid": args.beta_grid},
        "train_summary": train_summary,
        "seeds": seed_results,
        "means": {
            "latent_top1": float(np.mean([r["latent_top1_success_rate"] for r in seed_results])),
            "state_top1": float(np.mean([r["state_top1_success_rate"] for r in seed_results])),
            "oracle": float(np.mean([r["oracle_topk_success_rate"] for r in seed_results])),
        },
    }
    for beta in args.beta_grid:
        summary["means"][f"mixed_beta_{beta}"] = float(
            np.mean([r["mixed"][str(beta)]["success_rate"] for r in seed_results])
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["means"], indent=2), flush=True)


if __name__ == "__main__":
    main()
