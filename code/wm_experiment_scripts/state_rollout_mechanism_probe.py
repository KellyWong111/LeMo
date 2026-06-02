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


def clone_prepared(prepared):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared.items()}


def pairwise_stats(x: torch.Tensor):
    x = x.detach().float().cpu()
    m = min(x.shape[0], 160)
    y = x[:m]
    d = torch.cdist(y, y, p=2)
    iu = torch.triu_indices(m, m, offset=1)
    vals = d[iu[0], iu[1]]
    cov = torch.cov(x.T)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    pr = (eig.sum() ** 2 / eig.square().sum()).item()
    return {
        "latent_norm_mean": float(x.norm(dim=-1).mean().item()),
        "pairwise_l2_mean": float(vals.mean().item()),
        "cov_trace": float(eig.sum().item()),
        "participation_ratio": float(pr),
    }


def action_sensitivity(model, prepared_base, candidates, sigma=0.15, repeats=3):
    """Measure rollout final-latent sensitivity to block-action perturbations."""
    num_envs = candidates.shape[0]
    sample_count = min(candidates.shape[1], 8)
    base_candidates = candidates[:, :sample_count].to("cuda").float()
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, sample_count)
    with torch.inference_mode():
        _ = model.get_cost(prepared, base_candidates)
        pred0 = prepared["predicted_emb"][:, :, -1, :].detach().float().cpu()
    ratios = []
    for _ in range(repeats):
        noise = torch.randn_like(base_candidates) * sigma
        perturbed = base_candidates + noise
        prepared2 = clone_prepared(prepared_base)
        prepared2 = base.expand_info_for_candidates(prepared2, num_envs, sample_count)
        with torch.inference_mode():
            _ = model.get_cost(prepared2, perturbed)
            pred2 = prepared2["predicted_emb"][:, :, -1, :].detach().float().cpu()
        dz = (pred2 - pred0).norm(dim=-1).reshape(-1)
        da = noise.detach().float().cpu().reshape(num_envs, sample_count, -1).norm(dim=-1).reshape(-1)
        ratios.append(dz / da.clamp_min(1e-8))
    ratio = torch.cat(ratios)
    return {"dz_da_ratio_mean": float(ratio.mean().item()), "dz_da_ratio_std": float(ratio.std().item())}


@torch.inference_mode()
def rollout_metrics(model, prepared_base, candidates):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    costs = model.get_cost(prepared, candidates.to("cuda")).detach().float().cpu().numpy()
    pred = prepared["predicted_emb"].detach().float().cpu()  # B,K,T,D
    final = pred[:, :, -1, :]
    # spread among candidate final embeddings, per state
    spread = []
    for row in final:
        d = torch.cdist(row, row)
        iu = torch.triu_indices(row.shape[0], row.shape[0], offset=1)
        spread.append(float(d[iu[0], iu[1]].mean().item()))
    # temporal movement within each predicted trajectory
    step = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).norm(dim=-1)
    return {
        "costs": costs,
        "rollout_final_spread_mean": float(np.mean(spread)),
        "rollout_step_norm_mean": float(step.mean().item()),
        "rollout_step_norm_std": float(step.std().item()),
        "final_pr": pairwise_stats(final.reshape(-1, final.shape[-1]))["participation_ratio"],
    }


def success_cost_metrics(labels: np.ndarray, costs: np.ndarray):
    gaps, aucs = [], []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) and len(neg):
            gaps.append(float(neg.mean() - pos.mean()))
            aucs.append(float((pos[:, None] < neg[None, :]).mean()))
    return {
        "cost_gap": None if not gaps else float(np.mean(gaps)),
        "pairwise_success_auc": None if not aucs else float(np.mean(aucs)),
    }


def evaluate_policy(name, policy, seed, args):
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = policy
    cfg.eval.num_eval = args.num_eval
    cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    OmegaConf.update(cfg, "plan_config.horizon", args.horizon, merge=True)
    OmegaConf.update(cfg, "plan_config.action_block", args.action_block, merge=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", args.receding_horizon, merge=True)

    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False))
    indices = valid_indices[picked]
    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared_base = base.make_eval_like_info(
        raw_info,
        {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)},
        process,
    )
    model = base.load_model(cfg, cache_dir=args.cache_dir)

    # encoder geometry on current start states
    encoded = model.encode({"pixels": prepared_base["pixels"].to("cuda")})
    enc_flat = encoded["emb"].reshape(-1, encoded["emb"].shape[-1]).detach().float().cpu()
    geom = pairwise_stats(enc_flat)

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
    geom.update(action_sensitivity(model, prepared_base, candidates))
    roll = rollout_metrics(model, prepared_base, candidates.float())
    del model
    torch.cuda.empty_cache()

    plans = candidates.numpy()
    labels = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(labels, axis=1)
    top1 = float(labels[:, 0].mean() * 100.0)
    oracle = float(labels.any(axis=1).mean() * 100.0)
    out = {
        "name": name,
        "policy": policy,
        "seed": seed,
        "top1_success": top1,
        "oracle_top30": oracle,
        "topk_cost_mean": float(np.mean(topk_costs.numpy())),
        **geom,
        **{k: v for k, v in roll.items() if k != "costs"},
        **success_cost_metrics(labels, roll["costs"]),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME", "/data1/jingyixi/.stable_worldmodel"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=4)
    args = parser.parse_args()
    policies = {
        "gate07_ep4": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4",
        "state_roll_l003_ep1": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1",
    }
    rows = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in [int(x) for x in args.seeds.split(",") if x]:
        for name, policy in policies.items():
            print(f"[RUN] {name} seed={seed}", flush=True)
            rows.append(evaluate_policy(name, policy, seed, args))
            out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
