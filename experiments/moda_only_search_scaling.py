from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base
from topk_oracle_pilot import get_multistart_topk_candidates


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "moda_only_search_scaling_20260529"
POLICY = (
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/"
    "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1"
)


class FixedPlanPolicy:
    def __init__(self, plans: np.ndarray):
        self.plans = plans
        self.action_buffer = None

    def set_env(self, env):
        self.env = env
        plans = self.plans.reshape(self.plans.shape[0], -1, self.plans.shape[-1])
        self.action_buffer = deque(plans.transpose(1, 0, 2), maxlen=plans.shape[1])

    def get_action(self, info_dict, **kwargs):
        if self.action_buffer and len(self.action_buffer) > 0:
            return self.action_buffer.popleft()
        return np.zeros((self.env.num_envs, self.plans.shape[-1]), dtype=np.float32)


def eval_topk_plans_batched(cfg, dataset, process, indices, plans_topk):
    """Evaluate all top-k candidates in one vectorized world call.

    plans_topk has shape (N, K, horizon, action_dim). The returned label matrix
    has shape (N, K). This avoids reinitializing the world once per rank.
    """
    n_eval, topk = plans_topk.shape[:2]
    flat_plans = plans_topk.reshape(n_eval * topk, *plans_topk.shape[2:])
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    rows = dataset.get_row_data(np.asarray(indices))
    eval_episodes = np.repeat(rows[col_name], topk)
    eval_start_idx = np.repeat(rows["step_idx"], topk)

    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.eval.num_eval = int(n_eval * topk)
    eval_cfg.world.num_envs = int(n_eval * topk)
    world = swm.World(**OmegaConf.to_container(eval_cfg.world, resolve=True), image_shape=(224, 224))
    action_shape = tuple(world.envs.action_space.shape)
    env_action_dim = int(action_shape[-1])
    if "action" in process:
        shaped = flat_plans.reshape(
            flat_plans.shape[0],
            flat_plans.shape[1],
            int(cfg.plan_config.action_block),
            env_action_dim,
        )
        flat = shaped.reshape(-1, env_action_dim)
        flat = process["action"].inverse_transform(flat)
        shaped = flat.reshape(shaped.shape)
    else:
        shaped = flat_plans.reshape(
            flat_plans.shape[0],
            flat_plans.shape[1],
            int(cfg.plan_config.action_block),
            env_action_dim,
        )

    policy = FixedPlanPolicy(shaped.astype(np.float32))
    world.set_policy(policy)
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=eval_cfg.eval.goal_offset_steps,
        eval_budget=eval_cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(eval_cfg.eval.get("callables"), resolve=True),
        save_video=False,
        video_path="/tmp/moda_only_search_scaling_videos",
    )
    return np.asarray(metrics["episode_successes"], dtype=bool).reshape(n_eval, topk)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-eval", type=int, default=20)
    ap.add_argument("--num-samples-list", default="150,300")
    ap.add_argument("--cem-steps-list", default="15,30")
    ap.add_argument("--restarts-list", default="1,2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = POLICY
    cfg.eval.num_eval = args.num_eval
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

    model = base.load_model(cfg, cache_dir=None)
    rows = []
    case_rows = []
    for num_samples in parse_int_list(args.num_samples_list):
        for cem_steps in parse_int_list(args.cem_steps_list):
            for restarts in parse_int_list(args.restarts_list):
                topk_candidates, topk_costs = get_multistart_topk_candidates(
                    model,
                    prepared_base,
                    action_dim=action_dim,
                    horizon=int(cfg.plan_config.horizon),
                    num_samples=num_samples,
                    topk=args.topk,
                    n_steps=cem_steps,
                    seed=args.seed,
                    restarts=restarts,
                )
                labels = eval_topk_plans_batched(
                    cfg,
                    dataset,
                    process,
                    indices,
                    topk_candidates.numpy(),
                )
                first_success = []
                for label_row in labels:
                    hits = np.nonzero(label_row)[0]
                    first_success.append(int(hits[0] + 1) if len(hits) else None)
                row = {
                    "num_eval": args.num_eval,
                    "num_samples": num_samples,
                    "cem_steps": cem_steps,
                    "restarts": restarts,
                    "topk": args.topk,
                    "top1_success": float(labels[:, 0].mean() * 100.0),
                    "top3_success": float(labels[:, : min(3, args.topk)].any(axis=1).mean() * 100.0),
                    "top5_success": float(labels[:, : min(5, args.topk)].any(axis=1).mean() * 100.0),
                    "top10_success": float(labels[:, : min(10, args.topk)].any(axis=1).mean() * 100.0),
                    "topk_success": float(labels.any(axis=1).mean() * 100.0),
                    "episodes_with_success": int(labels.any(axis=1).sum()),
                    "near_miss_count": int(((~labels[:, 0]) & labels.any(axis=1)).sum()),
                    "mean_top1_cost": float(topk_costs[:, 0].mean()),
                    "mean_topk_cost": float(topk_costs.mean()),
                }
                rows.append(row)
                for i, rank in enumerate(first_success):
                    if (not labels[i, 0]) and labels[i].any():
                        case_rows.append(
                            {
                                "num_samples": num_samples,
                                "cem_steps": cem_steps,
                                "restarts": restarts,
                                "eval_i": i,
                                "dataset_index": int(indices[i]),
                                "first_success_rank": rank,
                                "top1_cost": float(topk_costs[i, 0]),
                            }
                        )
                write_csv(out / "moda_only_search_scaling.csv", rows)
                write_csv(out / "moda_only_search_scaling_cases.csv", case_rows)

    del model
    torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r["top1_success"]) if rows else None
    report = {"settings": vars(args), "policy": POLICY, "indices": indices.tolist(), "rows": rows, "best": best}
    (out / "moda_only_search_scaling.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# MoDA-Only Candidate Generation / Search Scaling",
        "",
        "This is MoDA-only planner scaling. It changes search budget/restarts only; no bsl fallback, no fixed/harmed metric, and no world-model retraining.",
        "",
        "|samples|steps|restarts|top1|top3|top5|top10|topK|near-miss|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"|{row['num_samples']}|{row['cem_steps']}|{row['restarts']}|"
            f"{row['top1_success']:.2f}|{row['top3_success']:.2f}|{row['top5_success']:.2f}|"
            f"{row['top10_success']:.2f}|{row['topk_success']:.2f}|{row['near_miss_count']}|"
        )
    if best is not None:
        md += [
            "",
            "## Verdict",
            "",
            (
                f"Best MoDA-only top1 is {best['top1_success']:.2f} with "
                f"num_samples={best['num_samples']}, cem_steps={best['cem_steps']}, "
                f"restarts={best['restarts']}."
            ),
        ]
    (out / "moda_only_search_scaling.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_search_scaling.md").read_text())


if __name__ == "__main__":
    main()
