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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import analyze_cem_margin as base
from topk_oracle_pilot import eval_fixed_plans, get_multistart_topk_candidates


def action_features(plans: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Build cheap planner-side features for a candidate action sequence.

    plans: (num_eval, topk, horizon, action_dim)
    costs: (num_eval, topk)
    """
    n_eval, topk = plans.shape[:2]
    flat = plans.reshape(n_eval, topk, -1)
    diffs = np.diff(flat, axis=2)
    rank = np.broadcast_to(np.arange(topk, dtype=np.float32)[None, :], costs.shape)
    feats = [
        costs,
        rank / max(topk - 1, 1),
        np.linalg.norm(flat, axis=2),
        np.mean(np.abs(flat), axis=2),
        np.max(np.abs(flat), axis=2),
        np.std(flat, axis=2),
        np.linalg.norm(diffs, axis=2),
        np.mean(np.abs(diffs), axis=2),
    ]
    return np.stack(feats, axis=-1).astype(np.float32)


@torch.inference_mode()
def latent_rollout_features(model, prepared_base, candidates: torch.Tensor, costs: torch.Tensor):
    """Extract planning-facing latent features from frozen world-model rollouts."""
    num_envs, topk = candidates.shape[:2]
    prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    cand = candidates.to("cuda")
    model_cost = model.get_cost(prepared, cand).detach().float().cpu().numpy()

    pred = prepared["predicted_emb"].detach().float()
    goal = prepared["goal_emb"].detach().float()
    if goal.ndim == pred.ndim - 1:
        goal = goal.unsqueeze(1)
    if goal.shape[1] == 1 and pred.shape[1] != 1:
        goal = goal.expand(-1, pred.shape[1], -1, -1)
    goal_last = goal[..., -1:, :].expand_as(pred)
    diff = pred - goal_last
    dist_t = diff.square().sum(dim=-1).sqrt().cpu().numpy()

    final_pred = pred[:, :, -1, :].cpu().numpy()
    final_goal = goal_last[:, :, -1, :].cpu().numpy()
    final_diff = final_pred - final_goal
    init_diff = pred[:, :, 0, :] - goal_last[:, :, 0, :]
    init_dist = init_diff.square().sum(dim=-1).sqrt().cpu().numpy()
    final_dist = dist_t[:, :, -1]

    n_eval, topk = model_cost.shape
    rank = np.broadcast_to(np.arange(topk, dtype=np.float32)[None, :], model_cost.shape)
    step = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    step_norm = step.square().sum(dim=-1).sqrt().cpu().numpy()

    # The first 8 dimensions are the deliberately small critic input:
    # final/trajectory goal distance + latent movement consistency.
    summary8 = np.stack(
        [
            final_dist,
            dist_t.mean(axis=-1),
            dist_t.min(axis=-1),
            init_dist - final_dist,
            dist_t.min(axis=-1) - final_dist,
            step_norm.mean(axis=-1),
            step_norm.std(axis=-1),
            step_norm.max(axis=-1),
        ],
        axis=-1,
    )

    scalar = np.stack(
        [
            costs.numpy(),
            model_cost,
            rank / max(topk - 1, 1),
            final_dist,
            dist_t.mean(axis=-1),
            dist_t.min(axis=-1),
            dist_t.std(axis=-1),
            init_dist - final_dist,
        ],
        axis=-1,
    )

    # Keep the full final latent difference. This is intentionally a diagnostic
    # for a learned planning critic; if this helps, the missing signal is in
    # success-aligned latent readout rather than CEM sampling.
    return {
        "summary8": summary8.astype(np.float32),
        "scalar": scalar.astype(np.float32),
        "final_diff": final_diff.astype(np.float32),
        "full": np.concatenate([scalar, final_diff], axis=-1).astype(np.float32),
    }


def first_success_rank(successes: np.ndarray) -> list[int | None]:
    out = []
    for row in successes:
        hit = np.nonzero(row)[0]
        out.append(None if len(hit) == 0 else int(hit[0]))
    return out


def leave_one_episode_rerank(features: np.ndarray, labels: np.ndarray, costs: np.ndarray):
    n_eval, topk, feat_dim = features.shape
    chosen = []
    pred_scores = []
    feasible = labels.reshape(-1)
    if len(np.unique(feasible)) < 2:
        return {
            "success_rate": None,
            "chosen_rank": [],
            "episode_successes": [],
            "reason": "only_one_label_class",
        }

    for heldout in range(n_eval):
        train_mask = np.ones(n_eval, dtype=bool)
        train_mask[heldout] = False
        x_train = features[train_mask].reshape(-1, feat_dim)
        y_train = labels[train_mask].reshape(-1).astype(int)
        if len(np.unique(y_train)) < 2:
            # Fall back to model cost if the training fold has only one class.
            rank = int(np.argmin(costs[heldout]))
            score = np.zeros(topk, dtype=np.float32)
        else:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=heldout,
                ),
            )
            clf.fit(x_train, y_train)
            score = clf.predict_proba(features[heldout])[:, 1]
            # Prefer high predicted success; use lower model cost as tie-breaker.
            rank = int(np.lexsort((costs[heldout], -score))[0])
        chosen.append(rank)
        pred_scores.append(score.tolist())

    episode_successes = labels[np.arange(n_eval), np.asarray(chosen)].astype(bool)
    return {
        "success_rate": float(episode_successes.mean() * 100.0),
        "chosen_rank": [int(x) for x in chosen],
        "episode_successes": episode_successes.tolist(),
        "pred_success_scores": pred_scores,
    }


def leave_one_episode_mlp_rerank(features: np.ndarray, labels: np.ndarray, costs: np.ndarray):
    n_eval, topk, feat_dim = features.shape
    if len(np.unique(labels.reshape(-1))) < 2:
        return {
            "success_rate": None,
            "chosen_rank": [],
            "episode_successes": [],
            "reason": "only_one_label_class",
        }

    chosen = []
    pred_scores = []
    for heldout in range(n_eval):
        train_mask = np.ones(n_eval, dtype=bool)
        train_mask[heldout] = False
        x_train = features[train_mask].reshape(-1, feat_dim)
        y_train = labels[train_mask].reshape(-1).astype(int)
        if len(np.unique(y_train)) < 2:
            rank = int(np.argmin(costs[heldout]))
            score = np.zeros(topk, dtype=np.float32)
        else:
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=min(256, len(y_train)),
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=20,
                    random_state=heldout + 17,
                ),
            )
            clf.fit(x_train, y_train)
            score = clf.predict_proba(features[heldout])[:, 1]
            rank = int(np.lexsort((costs[heldout], -score))[0])
        chosen.append(rank)
        pred_scores.append(score.tolist())

    episode_successes = labels[np.arange(n_eval), np.asarray(chosen)].astype(bool)
    return {
        "success_rate": float(episode_successes.mean() * 100.0),
        "chosen_rank": [int(x) for x in chosen],
        "episode_successes": episode_successes.tolist(),
        "pred_success_scores": pred_scores,
    }


class PairwiseRanker(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def leave_one_episode_pairwise_mlp_rerank(
    features: np.ndarray,
    labels: np.ndarray,
    costs: np.ndarray,
    *,
    seed_offset: int = 0,
    max_pairs_per_episode: int = 128,
    epochs: int = 250,
):
    """Train a CEM-candidate-level ranking critic with leave-one-episode eval.

    The loss is only formed from within-episode success/failure pairs, matching
    the planner use case: among the same CEM top-k candidate set, score a
    successful rollout above failed rollouts.
    """
    n_eval, topk, feat_dim = features.shape
    if len(np.unique(labels.reshape(-1))) < 2:
        return {
            "success_rate": None,
            "chosen_rank": [],
            "episode_successes": [],
            "reason": "only_one_label_class",
        }

    chosen = []
    pred_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for heldout in range(n_eval):
        train_mask = np.ones(n_eval, dtype=bool)
        train_mask[heldout] = False
        train_features = features[train_mask]
        train_labels = labels[train_mask]

        pair_pos = []
        pair_neg = []
        rng = np.random.default_rng(seed_offset + heldout + 4096)
        for ep_feat, ep_label in zip(train_features, train_labels):
            pos_idx = np.nonzero(ep_label)[0]
            neg_idx = np.nonzero(~ep_label)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            pairs = [(p, n) for p in pos_idx for n in neg_idx]
            if len(pairs) > max_pairs_per_episode:
                take = rng.choice(len(pairs), size=max_pairs_per_episode, replace=False)
                pairs = [pairs[i] for i in take]
            for p, n in pairs:
                pair_pos.append(ep_feat[p])
                pair_neg.append(ep_feat[n])

        x_flat_train = train_features.reshape(-1, feat_dim)
        x_test = features[heldout]
        x_flat_train_std, x_test_std = _standardize_train_test(x_flat_train, x_test)
        if not pair_pos:
            rank = int(np.argmin(costs[heldout]))
            score = np.zeros(topk, dtype=np.float32)
        else:
            # Standardize pair tensors with the same stats as the candidate pool.
            mean = x_flat_train.mean(axis=0, keepdims=True)
            std = x_flat_train.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            pos = torch.as_tensor((np.asarray(pair_pos, dtype=np.float32) - mean) / std, device=device).float()
            neg = torch.as_tensor((np.asarray(pair_neg, dtype=np.float32) - mean) / std, device=device).float()
            x_test_t = torch.as_tensor(x_test_std, device=device).float()

            torch.manual_seed(seed_offset + heldout + 17)
            model = PairwiseRanker(feat_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            margin = 1.0
            batch_size = min(512, pos.shape[0])
            for _ in range(epochs):
                perm = torch.randperm(pos.shape[0], device=device)
                for start in range(0, pos.shape[0], batch_size):
                    idx = perm[start : start + batch_size]
                    s_pos = model(pos[idx])
                    s_neg = model(neg[idx])
                    rank_loss = torch.nn.functional.softplus(margin - (s_pos - s_neg)).mean()
                    opt.zero_grad(set_to_none=True)
                    rank_loss.backward()
                    opt.step()
            with torch.inference_mode():
                score = model(x_test_t).detach().cpu().numpy()
            rank = int(np.lexsort((costs[heldout], -score))[0])
        chosen.append(rank)
        pred_scores.append(score.tolist())

    episode_successes = labels[np.arange(n_eval), np.asarray(chosen)].astype(bool)
    return {
        "success_rate": float(episode_successes.mean() * 100.0),
        "chosen_rank": [int(x) for x in chosen],
        "episode_successes": episode_successes.tolist(),
        "pred_success_scores": pred_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    args = parser.parse_args()

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

    plans = candidates.numpy()
    costs = topk_costs.numpy()
    top1_metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, 0])

    candidate_success = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        candidate_success.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(candidate_success, axis=1)

    action_feat = action_features(plans, costs)
    latent_feats = latent_rollout_features(model, prepared_base, candidates, topk_costs)
    del model
    torch.cuda.empty_cache()
    latent_feat = latent_feats["full"]
    summary8_feat = latent_feats["summary8"]
    scalar_feat = latent_feats["scalar"]
    combined_feat = np.concatenate([action_feat, latent_feat], axis=-1)
    rerank_action = leave_one_episode_rerank(action_feat, labels, costs)
    rerank_latent = leave_one_episode_rerank(latent_feat, labels, costs)
    rerank_combined = leave_one_episode_rerank(combined_feat, labels, costs)
    mlp_summary8 = leave_one_episode_mlp_rerank(summary8_feat, labels, costs)
    mlp_scalar = leave_one_episode_mlp_rerank(scalar_feat, labels, costs)
    mlp_full = leave_one_episode_mlp_rerank(latent_feat, labels, costs)
    cem_rank_summary8 = leave_one_episode_pairwise_mlp_rerank(summary8_feat, labels, costs)
    cem_rank_scalar = leave_one_episode_pairwise_mlp_rerank(scalar_feat, labels, costs, seed_offset=100)
    cem_rank_full = leave_one_episode_pairwise_mlp_rerank(latent_feat, labels, costs, seed_offset=200)
    cem_rank_combined = leave_one_episode_pairwise_mlp_rerank(combined_feat, labels, costs, seed_offset=300)
    oracle_any = labels.any(axis=1)

    result = {
        "policy": args.policy,
        "indices": indices.tolist(),
        "settings": vars(args),
        "top1_success_rate": float(np.mean(top1_metrics["episode_successes"]) * 100.0),
        "oracle_topk_success_rate": float(np.mean(oracle_any) * 100.0),
        "oracle_first_success_rank": first_success_rank(labels),
        "success_aligned_rerank": rerank_combined,
        "success_aligned_rerank_action_only": rerank_action,
        "success_aligned_rerank_latent_only": rerank_latent,
        "mlp_success_critic_summary8": mlp_summary8,
        "mlp_success_critic_scalar": mlp_scalar,
        "mlp_success_critic_full_latent": mlp_full,
        "cem_aware_pairwise_rank_summary8": cem_rank_summary8,
        "cem_aware_pairwise_rank_scalar": cem_rank_scalar,
        "cem_aware_pairwise_rank_full_latent": cem_rank_full,
        "cem_aware_pairwise_rank_combined": cem_rank_combined,
        "cost_success_diagnostic": {
            "mean_cost_success": None if not labels.any() else float(costs[labels].mean()),
            "mean_cost_failure": None if labels.all() else float(costs[~labels].mean()),
            "failure_minus_success": None
            if (not labels.any() or labels.all())
            else float(costs[~labels].mean() - costs[labels].mean()),
        },
        "top1_episode_successes": np.asarray(top1_metrics["episode_successes"], dtype=bool).tolist(),
        "oracle_episode_successes": oracle_any.tolist(),
        "candidate_successes_by_rank": [labels[:, rank].tolist() for rank in range(args.topk)],
        "topk_costs": costs.tolist(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
