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
POOL = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
OUT = ROOT / "moda_only_residual_proposal_20260529"
POLICY = (
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/"
    "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1"
)
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


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
    world.set_policy(FixedPlanPolicy(shaped.astype(np.float32)))
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=eval_cfg.eval.goal_offset_steps,
        eval_budget=eval_cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(eval_cfg.eval.get("callables"), resolve=True),
        save_video=False,
        video_path="/tmp/moda_only_residual_proposal_videos",
    )
    return np.asarray(metrics["episode_successes"], dtype=bool).reshape(n_eval, topk)


def model_cost_for_candidates(model, prepared_base, candidates: torch.Tensor) -> torch.Tensor:
    num_envs, num_samples = candidates.shape[:2]
    prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
    prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
    return model.get_cost(prepared, candidates)


def load_pool(seed: int):
    return np.load(POOL / f"vf05_mix20_seed{seed}.npz", allow_pickle=True)


def build_residual_bank(train_seeds: list[int], mode: str) -> np.ndarray:
    residuals = []
    for seed in train_seeds:
        data = load_pool(seed)
        actions = data["actions"].astype(np.float32)
        costs = data["costs"].astype(np.float32)
        labels = data["labels"].astype(bool)
        for ep in range(actions.shape[0]):
            rank0 = int(np.argmin(costs[ep]))
            success = np.nonzero(labels[ep])[0]
            if len(success) == 0 or labels[ep, rank0]:
                continue
            if mode == "best_success":
                chosen = [int(success[np.argmin(costs[ep, success])])]
            elif mode == "all_success":
                chosen = [int(x) for x in success]
            else:
                raise ValueError(f"unknown residual mode: {mode}")
            for pos in chosen:
                residuals.append(actions[ep, pos] - actions[ep, rank0])
    if not residuals:
        raise RuntimeError("empty residual bank")
    return np.stack(residuals).astype(np.float32)


def select_prototypes(bank: np.ndarray, num_prototypes: int, seed: int) -> np.ndarray:
    flat = bank.reshape(bank.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    prototypes = [mean.reshape(bank.shape[1:])]
    norms = np.linalg.norm(centered, axis=1)
    if len(bank) > 1:
        prototypes.append(bank[int(np.argmin(norms))])
        prototypes.append(bank[int(np.argmax(norms))])
    # Farthest-point sampling gives diverse residual directions without needing sklearn.
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(bank)))
    chosen = [start]
    dists = np.linalg.norm(flat - flat[start], axis=1)
    while len(chosen) < min(num_prototypes, len(bank)):
        idx = int(np.argmax(dists))
        chosen.append(idx)
        dists = np.minimum(dists, np.linalg.norm(flat - flat[idx], axis=1))
    prototypes.extend([bank[i] for i in chosen])
    proto = np.stack(prototypes).astype(np.float32)
    # Deduplicate near-identical prototypes.
    uniq = []
    seen = set()
    for p in proto:
        key = tuple(np.round(p.reshape(-1), 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return np.stack(uniq[:num_prototypes]).astype(np.float32)


def make_residual_candidates(raw_topk: torch.Tensor, prototypes: np.ndarray, scales: list[float], base_top: int):
    device = raw_topk.device
    bases = raw_topk[:, : min(base_top, raw_topk.shape[1])]
    aligned = np.zeros(tuple([prototypes.shape[0], *raw_topk.shape[2:]]), dtype=np.float32)
    h = min(aligned.shape[1], prototypes.shape[1])
    d = min(aligned.shape[2], prototypes.shape[2])
    aligned[:, :h, :d] = prototypes[:, :h, :d]
    proto = torch.tensor(aligned, device=device, dtype=raw_topk.dtype)
    generated = [raw_topk]
    for scale in scales:
        shifted = bases[:, :, None] + float(scale) * proto[None, None]
        shifted = shifted.reshape(raw_topk.shape[0], -1, *raw_topk.shape[2:])
        generated.append(shifted)
    return torch.cat(generated, dim=1)


def metrics_from_labels(labels: np.ndarray, row_extra: dict):
    topk = labels.shape[1]
    row = dict(row_extra)
    row.update(
        {
            "top1_success": float(labels[:, 0].mean() * 100.0),
            "top3_success": float(labels[:, : min(3, topk)].any(axis=1).mean() * 100.0),
            "top5_success": float(labels[:, : min(5, topk)].any(axis=1).mean() * 100.0),
            "top10_success": float(labels[:, : min(10, topk)].any(axis=1).mean() * 100.0),
            "topk_success": float(labels.any(axis=1).mean() * 100.0),
            "episodes_with_success": int(labels.any(axis=1).sum()),
            "near_miss_count": int(((~labels[:, 0]) & labels.any(axis=1)).sum()),
        }
    )
    return row


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


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-eval", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--cem-steps", type=int, default=15)
    ap.add_argument("--raw-topk", type=int, default=10)
    ap.add_argument("--eval-topk", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--base-top-list", default="1,3,5")
    ap.add_argument("--scales", default="0.25,0.5,1.0,1.5")
    ap.add_argument("--num-prototypes", type=int, default=12)
    ap.add_argument("--residual-mode", choices=["best_success", "all_success"], default="best_success")
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

    scales = parse_float_list(args.scales)
    base_tops = [int(x) for x in args.base_top_list.split(",") if x.strip()]
    rows = []
    case_rows = []
    model = base.load_model(cfg, cache_dir=None)
    for split, (train_seeds, _val_seeds) in SPLITS.items():
        bank = build_residual_bank(train_seeds, args.residual_mode)
        prototypes = select_prototypes(bank, args.num_prototypes, args.seed)
        raw_topk, raw_costs = get_multistart_topk_candidates(
            model,
            prepared_base,
            action_dim=action_dim,
            horizon=int(cfg.plan_config.horizon),
            num_samples=args.num_samples,
            topk=args.raw_topk,
            n_steps=args.cem_steps,
            seed=args.seed,
            restarts=args.restarts,
        )
        raw_labels = eval_topk_plans_batched(cfg, dataset, process, indices, raw_topk[:, : args.eval_topk].numpy())
        rows.append(
            metrics_from_labels(
                raw_labels,
                {
                    "split": split,
                    "method": "raw_cem",
                    "num_eval": args.num_eval,
                    "num_samples": args.num_samples,
                    "cem_steps": args.cem_steps,
                    "restarts": args.restarts,
                    "raw_topk": args.raw_topk,
                    "eval_topk": args.eval_topk,
                    "base_top": 0,
                    "num_bank": int(len(bank)),
                    "num_prototypes": int(len(prototypes)),
                    "scales": "",
                    "mean_top1_cost": float(raw_costs[:, 0].mean()),
                },
            )
        )
        for base_top in base_tops:
            candidate_pool = make_residual_candidates(raw_topk.cuda(), prototypes, scales, base_top)
            costs = model_cost_for_candidates(model, prepared_base, candidate_pool).detach().cpu()
            order = torch.argsort(costs, dim=1)[:, : args.eval_topk]
            batch = torch.arange(candidate_pool.shape[0])[:, None]
            selected = candidate_pool.detach().cpu()[batch, order]
            selected_costs = costs[batch, order]
            labels = eval_topk_plans_batched(cfg, dataset, process, indices, selected.numpy())
            row = metrics_from_labels(
                labels,
                {
                    "split": split,
                    "method": "residual_shift_rawcost",
                    "num_eval": args.num_eval,
                    "num_samples": args.num_samples,
                    "cem_steps": args.cem_steps,
                    "restarts": args.restarts,
                    "raw_topk": args.raw_topk,
                    "eval_topk": args.eval_topk,
                    "base_top": int(base_top),
                    "num_bank": int(len(bank)),
                    "num_prototypes": int(len(prototypes)),
                    "scales": args.scales,
                    "mean_top1_cost": float(selected_costs[:, 0].mean()),
                },
            )
            rows.append(row)
            for i in range(labels.shape[0]):
                if (not labels[i, 0]) and labels[i].any():
                    hits = np.nonzero(labels[i])[0]
                    case_rows.append(
                        {
                            "split": split,
                            "base_top": int(base_top),
                            "eval_i": int(i),
                            "dataset_index": int(indices[i]),
                            "first_success_rank": int(hits[0] + 1),
                            "top1_cost": float(selected_costs[i, 0]),
                        }
                    )
            write_csv(out / "moda_only_residual_proposal.csv", rows)
            write_csv(out / "moda_only_residual_proposal_cases.csv", case_rows)

    del model
    torch.cuda.empty_cache()
    agg = []
    keys = sorted({(r["method"], r["base_top"]) for r in rows})
    for method, base_top in keys:
        rs = [r for r in rows if r["method"] == method and r["base_top"] == base_top]
        agg.append(
            {
                "method": method,
                "base_top": base_top,
                "top1_success": float(np.mean([r["top1_success"] for r in rs])),
                "top3_success": float(np.mean([r["top3_success"] for r in rs])),
                "top5_success": float(np.mean([r["top5_success"] for r in rs])),
                "top10_success": float(np.mean([r["top10_success"] for r in rs])),
                "topk_success": float(np.mean([r["topk_success"] for r in rs])),
                "near_miss_count": int(sum(r["near_miss_count"] for r in rs)),
            }
        )
    best = max(agg, key=lambda r: r["top1_success"]) if agg else None
    write_csv(out / "moda_only_residual_proposal_aggregate.csv", agg)
    report = {"settings": vars(args), "policy": POLICY, "rows": rows, "aggregate": agg, "best": best}
    (out / "moda_only_residual_proposal.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# MoDA-Only Success-Conditioned Residual Proposal",
        "",
        "Residuals are learned from stateroll/MoDA candidates only: success action minus raw rank0 failure action. No bsl fallback, no fixed/harmed metric, no world-model retraining.",
        "",
        "|method|base_top|top1|top3|top5|top10|topK|near-miss|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        md.append(
            f"|{r['method']}|{r['base_top']}|{r['top1_success']:.2f}|{r['top3_success']:.2f}|"
            f"{r['top5_success']:.2f}|{r['top10_success']:.2f}|{r['topk_success']:.2f}|{r['near_miss_count']}|"
        )
    if best:
        md += [
            "",
            "## Verdict",
            "",
            f"Best MoDA-only top1 is {best['top1_success']:.2f} with method={best['method']} base_top={best['base_top']}.",
        ]
    (out / "moda_only_residual_proposal.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_residual_proposal.md").read_text())


if __name__ == "__main__":
    main()
