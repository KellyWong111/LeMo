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


@torch.inference_mode()
def score_candidates(model, prepared_base, candidates: torch.Tensor):
    num_envs, topk = candidates.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, num_envs, topk)
    return model.get_cost(prepared, candidates.to("cuda").float()).detach().float().cpu().numpy()


def success_rate_for_argmin(labels: np.ndarray, costs: np.ndarray):
    pick = np.argmin(costs, axis=1)
    return float(labels[np.arange(labels.shape[0]), pick].mean() * 100.0), pick.tolist()


def rowwise_pairwise_auc(labels: np.ndarray, costs: np.ndarray):
    aucs = []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) and len(neg):
            aucs.append(float((pos[:, None] < neg[None, :]).mean()))
    return None if not aucs else float(np.mean(aucs))


def cost_gap(labels: np.ndarray, costs: np.ndarray):
    gaps = []
    for row_labels, row_costs in zip(labels, costs):
        pos = row_costs[row_labels]
        neg = row_costs[~row_labels]
        if len(pos) and len(neg):
            gaps.append(float(neg.mean() - pos.mean()))
    return None if not gaps else float(np.mean(gaps))


def rank_corr_row(a: np.ndarray, b: np.ndarray):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    if ar.std() < 1e-8 or br.std() < 1e-8:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def cost_stats_for_row(labels: np.ndarray, costs: np.ndarray):
    pos = costs[labels]
    neg = costs[~labels]
    return {
        "success_min": None if len(pos) == 0 else float(pos.min()),
        "success_mean": None if len(pos) == 0 else float(pos.mean()),
        "failure_min": None if len(neg) == 0 else float(neg.min()),
        "failure_mean": None if len(neg) == 0 else float(neg.mean()),
        "gap_failure_minus_success": None
        if len(pos) == 0 or len(neg) == 0
        else float(neg.mean() - pos.mean()),
    }


def top_ranks(costs: np.ndarray, labels: np.ndarray, k: int = 5):
    order = np.argsort(costs)[:k]
    return [
        {
            "rank": int(rank),
            "cost": float(costs[rank]),
            "success": bool(labels[rank]),
        }
        for rank in order
    ]


def build_episode_debug(
    indices: np.ndarray,
    labels: np.ndarray,
    candidate_costs: np.ndarray,
    teacher_costs: np.ndarray,
    candidate_pick: list[int],
    teacher_pick: list[int],
    first_success_rank: list[int | None],
):
    rows = []
    for env_idx in range(labels.shape[0]):
        cand_rank = int(candidate_pick[env_idx])
        teacher_rank = int(teacher_pick[env_idx])
        row_labels = labels[env_idx]
        row_candidate_costs = candidate_costs[env_idx]
        row_teacher_costs = teacher_costs[env_idx]
        rows.append(
            {
                "env_idx": int(env_idx),
                "dataset_index": int(indices[env_idx]),
                "oracle_success_exists": bool(row_labels.any()),
                "first_success_rank": first_success_rank[env_idx],
                "candidate_selected_rank": cand_rank,
                "candidate_selected_cost": float(row_candidate_costs[cand_rank]),
                "candidate_selected_success": bool(row_labels[cand_rank]),
                "official_selected_rank": teacher_rank,
                "official_selected_cost": float(row_teacher_costs[teacher_rank]),
                "official_selected_success": bool(row_labels[teacher_rank]),
                "top5_candidate_cost_ranks": top_ranks(row_candidate_costs, row_labels, k=5),
                "top5_official_cost_ranks": top_ranks(row_teacher_costs, row_labels, k=5),
                "candidate_cost_stats": cost_stats_for_row(row_labels, row_candidate_costs),
                "official_cost_stats": cost_stats_for_row(row_labels, row_teacher_costs),
            }
        )
    return rows


def evaluate_pair(name: str, candidate_policy: str, teacher_policy: str, seed: int, args):
    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = candidate_policy
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

    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)

    candidate_model = base.load_model(cfg, cache_dir=args.cache_dir)
    candidates, candidate_topk_costs = get_multistart_topk_candidates(
        candidate_model,
        prepared_base,
        action_dim=action_dim,
        horizon=int(cfg.plan_config.horizon),
        num_samples=args.num_samples,
        topk=args.topk,
        n_steps=args.cem_steps,
        seed=seed,
        restarts=args.restarts,
    )
    candidate_costs = score_candidates(candidate_model, prepared_base, candidates.float())
    del candidate_model
    torch.cuda.empty_cache()

    cfg.policy = teacher_policy
    teacher_model = base.load_model(cfg, cache_dir=args.cache_dir)
    teacher_costs = score_candidates(teacher_model, prepared_base, candidates.float())
    del teacher_model
    torch.cuda.empty_cache()

    plans = candidates.numpy()
    labels = []
    for rank in range(args.topk):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, plans[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=bool))
    labels = np.stack(labels, axis=1)

    cand_top1 = float(labels[:, 0].mean() * 100.0)
    teacher_top1, teacher_pick = success_rate_for_argmin(labels, teacher_costs)
    candidate_argmin, candidate_pick = success_rate_for_argmin(labels, candidate_costs)
    oracle = float(labels.any(axis=1).mean() * 100.0)
    first_success_rank = []
    for row in labels:
        hits = np.nonzero(row)[0]
        first_success_rank.append(int(hits[0]) if len(hits) else None)

    rank_corrs = [rank_corr_row(c, t) for c, t in zip(candidate_costs, teacher_costs)]
    rank_corrs = [x for x in rank_corrs if x is not None]

    return {
        "name": name,
        "seed": seed,
        "candidate_policy": candidate_policy,
        "teacher_policy": teacher_policy,
        "settings": {
            "num_eval": args.num_eval,
            "topk": args.topk,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "restarts": args.restarts,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "receding_horizon": args.receding_horizon,
        },
        "candidate_cost_top1_success": cand_top1,
        "candidate_cost_argmin_success": candidate_argmin,
        "official_teacher_argmin_success": teacher_top1,
        "oracle_topk_success": oracle,
        "teacher_gain_vs_candidate_top1": teacher_top1 - cand_top1,
        "teacher_oracle_gap": oracle - teacher_top1,
        "candidate_oracle_gap": oracle - cand_top1,
        "candidate_success_auc": rowwise_pairwise_auc(labels, candidate_costs),
        "teacher_success_auc": rowwise_pairwise_auc(labels, teacher_costs),
        "candidate_cost_gap": cost_gap(labels, candidate_costs),
        "teacher_cost_gap": cost_gap(labels, teacher_costs),
        "candidate_teacher_rank_corr_mean": None if not rank_corrs else float(np.mean(rank_corrs)),
        "first_success_rank": first_success_rank,
        "candidate_pick_by_model_cost": candidate_pick,
        "teacher_pick_by_official_cost": teacher_pick,
        "episode_debug": build_episode_debug(
            indices,
            labels,
            candidate_costs,
            teacher_costs,
            candidate_pick,
            teacher_pick,
            first_success_rank,
        ),
        "candidate_successes_by_rank": labels.T.tolist(),
        "candidate_costs": candidate_costs.tolist(),
        "official_teacher_costs": teacher_costs.tolist(),
        "indices": indices.tolist(),
    }


def summarize(rows):
    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    lines = [
        "# Official-guided rerank diagnostic",
        "",
        "|candidate_pool|seeds|candidate_top1|official_rerank|oracle|teacher_gain|teacher_auc|candidate_auc|teacher_gap|candidate_gap|rank_corr|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def avg(items, key):
        vals = [x.get(key) for x in items if x.get(key) is not None]
        return None if not vals else float(np.mean(vals))
    def fmt(x):
        return "NA" if x is None else f"{x:.1f}"
    def fmt3(x):
        return "NA" if x is None else f"{x:.3f}"
    for name, items in sorted(by_name.items()):
        lines.append(
            f"|{name}|{len(items)}|"
            f"{fmt(avg(items, 'candidate_cost_top1_success'))}|"
            f"{fmt(avg(items, 'official_teacher_argmin_success'))}|"
            f"{fmt(avg(items, 'oracle_topk_success'))}|"
            f"{fmt(avg(items, 'teacher_gain_vs_candidate_top1'))}|"
            f"{fmt3(avg(items, 'teacher_success_auc'))}|"
            f"{fmt3(avg(items, 'candidate_success_auc'))}|"
            f"{fmt3(avg(items, 'teacher_cost_gap'))}|"
            f"{fmt3(avg(items, 'candidate_cost_gap'))}|"
            f"{fmt3(avg(items, 'candidate_teacher_rank_corr_mean'))}|"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument(
        "--pools",
        default="gate07_ep4_pool,stateroll_l003_ep1_pool",
        help="Comma-separated pool names to run.",
    )
    args = parser.parse_args()

    teacher = "pusht_official_clean_5090_gpu0/lewm_pusht_official_clean_epoch_13"
    pools = {
        "gate07_ep4_pool": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_epoch_4",
        "stateroll_l003_ep1_pool": "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1",
    }
    pool_names = [x for x in args.pools.split(",") if x]
    pools = {name: pools[name] for name in pool_names}
    seeds = [int(x) for x in args.seeds.split(",") if x]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_path = outdir / "raw_results.json"
    for seed in seeds:
        for name, policy in pools.items():
            print(f"[RUN] {name} seed={seed}", flush=True)
            row = evaluate_pair(name, policy, teacher, seed, args)
            rows.append(row)
            raw_path.write_text(json.dumps(rows, indent=2))
            (outdir / "summary.md").write_text(summarize(rows))
    print(summarize(rows))


if __name__ == "__main__":
    main()
