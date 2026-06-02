from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base
import moda_only_calibrated_cem as cal
from moda_only_search_scaling import eval_topk_plans_batched
from topk_oracle_pilot import get_multistart_topk_candidates


ROOT = Path("/data1/jingyixi/wm_runs")
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
OUT = ROOT / "moda_only_learned_residual_proposal_20260530"
POLICY = (
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/"
    "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1"
)
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def goal_for_pred(goal: np.ndarray, pred: np.ndarray) -> np.ndarray:
    g = goal
    if g.ndim == 2:
        g = g[:, None, :]
    if g.ndim == 3 and pred.ndim == 4:
        g = g[:, None, :, :]
    if g.shape[1] == 1:
        g = np.repeat(g, pred.shape[1], axis=1)
    if g.shape[2] == 1:
        g = np.repeat(g, pred.shape[2], axis=2)
    elif g.shape[2] != pred.shape[2]:
        g = g[:, :, -pred.shape[2] :, :]
    return g


def pool_stats(pred: np.ndarray, goal: np.ndarray, actions: np.ndarray) -> dict:
    dist = np.sqrt(((pred - goal_for_pred(goal, pred)) ** 2).sum(axis=-1))
    anorm = np.sqrt((actions**2).sum(axis=-1))
    return {
        "final": dist[:, :, -1],
        "mean": dist.mean(axis=2),
        "min": dist.min(axis=2),
        "progress": dist[:, :, 0] - dist[:, :, -1],
        "latent_mean": pred.mean(axis=(2, 3)),
        "latent_std": pred.std(axis=(2, 3)),
        "action_norm": anorm.mean(axis=2),
        "action_std": anorm.std(axis=2),
    }


def feature_np(costs: np.ndarray, stats: dict, actions: np.ndarray, ep: int, cand: int) -> np.ndarray:
    c = costs[ep]
    order = np.argsort(c, kind="stable")
    ranks = np.empty(len(c), dtype=np.float64)
    ranks[order] = np.arange(len(c))
    sorted_c = c[order]
    base = [
        1.0,
        float(c[cand]),
        float(-c[cand]),
        float(ranks[cand] / max(1, len(c) - 1)),
        float((c[cand] - c.mean()) / (c.std() + 1e-6)),
        float(c[cand] - sorted_c[0]),
        float(c[cand] - sorted_c[min(4, len(c) - 1)]),
        float(sorted_c[1] - sorted_c[0]) if len(c) > 1 else 0.0,
        float(sorted_c[min(4, len(c) - 1)] - sorted_c[0]),
        float(c.std()),
        float(stats["final"][ep, cand]),
        float(stats["mean"][ep, cand]),
        float(stats["min"][ep, cand]),
        float(stats["progress"][ep, cand]),
        float(stats["latent_mean"][ep, cand]),
        float(stats["latent_std"][ep, cand]),
        float(stats["action_norm"][ep, cand]),
        float(stats["action_std"][ep, cand]),
    ]
    return np.asarray(base + actions[ep, cand].reshape(-1).tolist(), dtype=np.float64)


def train_data(train_seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seed in train_seeds:
        a = np.load(ST_ACTION / f"vf05_mix20_seed{seed}.npz", allow_pickle=True)
        r = np.load(ST_RAW / f"vf05_mix20_seed{seed}.npz", allow_pickle=True)
        costs = a["costs"].astype(np.float64)
        labels = a["labels"].astype(bool)
        actions = a["actions"].astype(np.float64)
        stats = pool_stats(r["pred"].astype(np.float64), r["goal"].astype(np.float64), actions)
        for ep in range(labels.shape[0]):
            rank0 = int(np.argmin(costs[ep]))
            success = np.nonzero(labels[ep])[0]
            if len(success) == 0 or labels[ep, rank0]:
                continue
            best_success = int(success[np.argmin(costs[ep, success])])
            xs.append(feature_np(costs, stats, actions, ep, rank0))
            ys.append((actions[ep, best_success] - actions[ep, rank0]).reshape(-1))
    if not xs:
        raise RuntimeError("no residual training examples")
    return np.stack(xs), np.stack(ys)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 10.0) -> dict:
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mean) / std
    z[:, 0] = 1.0
    a = z.T @ z + ridge * np.eye(z.shape[1])
    a[0, 0] -= ridge
    w = np.linalg.solve(a, z.T @ y)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "w": w.astype(np.float32)}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    target_dim = int(model["mean"].shape[0])
    if x.shape[1] > target_dim:
        x = x[:, :target_dim]
    elif x.shape[1] < target_dim:
        pad = np.zeros((x.shape[0], target_dim - x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)
    z = (x - model["mean"]) / model["std"]
    z[:, 0] = 1.0
    return z @ model["w"]


def online_feature(raw_cost: torch.Tensor, pred: torch.Tensor, goal_emb: torch.Tensor, actions: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        u = cal.utility_score_torch(raw_cost, pred, goal_emb, actions, {"mean": np.zeros(16, np.float32), "std": np.ones(16, np.float32), "w": np.zeros(16, np.float32)})
        # Recompute the same first 18 scalar features without relying on utility weights.
        goal = goal_emb
        if goal.ndim == pred.ndim - 1:
            goal = goal.unsqueeze(1)
        if goal.shape[1] == 1 and pred.shape[1] != 1:
            goal = goal.expand(-1, pred.shape[1], -1, -1)
        goal = goal[..., -1:, :].expand_as(pred)
        dist = torch.sqrt(((pred - goal) ** 2).sum(dim=-1) + 1e-12)
        anorm = torch.sqrt((actions**2).sum(dim=-1) + 1e-12)
        c = raw_cost
        order = torch.argsort(c, dim=1, stable=True)
        ranks = torch.empty_like(order)
        ar = torch.arange(c.shape[1], device=c.device)[None, :].expand_as(order)
        ranks.scatter_(1, order, ar)
        sorted_c = torch.gather(c, 1, order)
        feat = torch.stack(
            [
                torch.ones_like(c),
                c,
                -c,
                ranks.float() / max(1, c.shape[1] - 1),
                (c - c.mean(dim=1, keepdim=True)) / (c.std(dim=1, keepdim=True, unbiased=False) + 1e-6),
                c - sorted_c[:, 0:1],
                c - sorted_c[:, min(4, c.shape[1] - 1) : min(4, c.shape[1] - 1) + 1],
                (sorted_c[:, 1:2] - sorted_c[:, 0:1]).expand_as(c) if c.shape[1] > 1 else torch.zeros_like(c),
                (sorted_c[:, min(4, c.shape[1] - 1) : min(4, c.shape[1] - 1) + 1] - sorted_c[:, 0:1]).expand_as(c),
                c.std(dim=1, keepdim=True, unbiased=False).expand_as(c),
                dist[:, :, -1],
                dist.mean(dim=2),
                dist.min(dim=2).values,
                dist[:, :, 0] - dist[:, :, -1],
                pred.mean(dim=(2, 3)),
                pred.std(dim=(2, 3)),
                anorm.mean(dim=2),
                anorm.std(dim=2),
            ],
            dim=-1,
        )
        full = torch.cat([feat, actions.reshape(actions.shape[0], actions.shape[1], -1)], dim=-1)
        return full.detach().cpu().numpy()


@torch.inference_mode()
def score_candidates(model, prepared_base, candidates: torch.Tensor, util: dict, lamb: float):
    num_envs, num_samples = candidates.shape[:2]
    prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
    prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
    raw, pred, goal = cal.model_rollout_cost(model, prepared, candidates)
    u = cal.utility_score_torch(raw, pred, goal, candidates, util)
    return raw, raw - float(lamb) * u, u


def labels_to_row(labels: np.ndarray, row: dict) -> dict:
    out = dict(row)
    out.update(
        {
            "top1_success": float(labels[:, 0].mean() * 100),
            "top3_success": float(labels[:, : min(3, labels.shape[1])].any(axis=1).mean() * 100),
            "top5_success": float(labels[:, : min(5, labels.shape[1])].any(axis=1).mean() * 100),
            "oracle": float(labels.any(axis=1).mean() * 100),
            "success_density": float(labels.mean() * 100),
            "near_miss_count": int(((~labels[:, 0]) & labels.any(axis=1)).sum()),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_split(split: str, train_seeds: list[int], args) -> list[dict]:
    x, y = train_data(train_seeds)
    residual = fit_ridge(x, y, ridge=args.ridge)
    util = cal.fit_utility(train_seeds)
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = POLICY
    cfg.eval.num_eval = args.num_eval
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(args.seed)
    indices = valid_indices[np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))]
    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared_base = base.make_eval_like_info(raw_info, {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)}, process)
    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)
    wm = base.load_model(cfg, cache_dir=None)
    raw_topk, raw_costs = get_multistart_topk_candidates(
        wm, prepared_base, action_dim, int(cfg.plan_config.horizon), args.num_samples, args.raw_topk, args.cem_steps, args.seed, args.restarts
    )
    rows = []
    raw_labels = eval_topk_plans_batched(cfg, dataset, process, indices, raw_topk[:, : args.eval_topk].numpy())
    rows.append(labels_to_row(raw_labels, {"split": split, "method": "raw_moda", "lambda": 0.0, "scale": 0.0}))

    prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
    prepared = base.expand_info_for_candidates(prepared, raw_topk.shape[0], raw_topk.shape[1])
    raw2, pred, goal = cal.model_rollout_cost(wm, prepared, raw_topk.cuda())
    feat = online_feature(raw2, pred, goal, raw_topk.cuda())[:, 0]
    delta_small = predict_ridge(residual, feat).reshape(args.num_eval, 4, 10)
    delta = np.zeros((args.num_eval, *raw_topk.shape[2:]), dtype=np.float32)
    h = min(delta.shape[1], delta_small.shape[1])
    d = min(delta.shape[2], delta_small.shape[2])
    delta[:, :h, :d] = delta_small[:, :h, :d]
    delta_t = torch.tensor(delta, dtype=raw_topk.dtype)
    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    for scale in scales:
        shifted = raw_topk[:, : args.base_top] + float(scale) * delta_t[:, None]
        pool = torch.cat([raw_topk, shifted], dim=1).cuda()
        raw_s, plan_s, _u = score_candidates(wm, prepared_base, pool, util, args.cal_lambda)
        for method, score in [("residual_raw_cost", -raw_s), ("residual_calibrated_cost", -plan_s)]:
            order = torch.argsort(score, dim=1, descending=True)[:, : args.eval_topk]
            batch = torch.arange(pool.shape[0], device=pool.device)[:, None]
            selected = pool[batch, order].detach().cpu()
            labels = eval_topk_plans_batched(cfg, dataset, process, indices, selected.numpy())
            rows.append(labels_to_row(labels, {"split": split, "method": method, "lambda": args.cal_lambda, "scale": scale}))
    del wm
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-eval", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--cem-steps", type=int, default=15)
    ap.add_argument("--raw-topk", type=int, default=10)
    ap.add_argument("--eval-topk", type=int, default=10)
    ap.add_argument("--base-top", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--scales", default="0.25,0.5,1.0,1.5")
    ap.add_argument("--cal-lambda", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (train, _val) in SPLITS.items():
        rows.extend(run_split(split, train, args))
        write_csv(out / "moda_only_learned_residual_proposal.csv", rows)
    agg = []
    for key in sorted({(r["method"], r["scale"]) for r in rows}):
        rs = [r for r in rows if (r["method"], r["scale"]) == key]
        agg.append({k: key[i] for i, k in enumerate(["method", "scale"])} | {
            "top1_success": float(np.mean([r["top1_success"] for r in rs])),
            "oracle": float(np.mean([r["oracle"] for r in rs])),
            "success_density": float(np.mean([r["success_density"] for r in rs])),
            "near_miss_count": int(sum(r["near_miss_count"] for r in rs)),
        })
    best = max(agg, key=lambda r: (r["top1_success"], r["oracle"]))
    write_csv(out / "moda_only_learned_residual_proposal_aggregate.csv", agg)
    (out / "moda_only_learned_residual_proposal.json").write_text(json.dumps({"settings": vars(args), "rows": rows, "aggregate": agg, "best": best}, indent=2) + "\n")
    md = ["# MoDA-Only Learned Residual Proposal", "", "|method|scale|top1|oracle|success density|near-miss|", "|---|---:|---:|---:|---:|---:|"]
    for r in agg:
        md.append(f"|{r['method']}|{r['scale']}|{r['top1_success']:.2f}|{r['oracle']:.2f}|{r['success_density']:.2f}|{r['near_miss_count']}|")
    md += ["", "## Verdict", "", f"Best top1 is {best['top1_success']:.2f} with method={best['method']} scale={best['scale']}."]
    (out / "moda_only_learned_residual_proposal.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_learned_residual_proposal.md").read_text())


if __name__ == "__main__":
    main()
