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


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "moda_only_planner_in_loop_calibrated_cem_20260530"
POLICY = (
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/"
    "lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1"
)
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def row_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    pos = scores[labels]
    neg = scores[~labels]
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return float(wins / total)


def candidate_diversity(cands: torch.Tensor) -> float:
    flat = cands.reshape(cands.shape[0], cands.shape[1], -1).detach().cpu().numpy()
    return float(flat.std(axis=1).mean())


@torch.inference_mode()
def generate_candidates(
    model,
    prepared_base,
    action_dim: int,
    horizon: int,
    num_samples: int,
    topk: int,
    n_steps: int,
    seed: int,
    util: dict,
    lamb: float,
    mode: str,
):
    device = "cuda"
    num_envs = next(v for v in prepared_base.values() if torch.is_tensor(v)).shape[0]
    mean = torch.zeros(num_envs, horizon, action_dim, device=device)
    var = torch.ones(num_envs, horizon, action_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    final_cand = final_raw = final_util = final_plan = None
    for _ in range(n_steps):
        candidates = torch.randn(num_envs, num_samples, horizon, action_dim, generator=gen, device=device)
        candidates = candidates * var[:, None] + mean[:, None]
        candidates[:, 0] = mean
        prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
        prepared = base.expand_info_for_candidates(prepared, num_envs, num_samples)
        raw_cost, pred, goal = cal.model_rollout_cost(model, prepared, candidates)
        util_score = cal.utility_score_torch(raw_cost, pred, goal, candidates, util)
        plan_cost = raw_cost - float(lamb) * util_score
        elite_cost = plan_cost if mode == "planner_in_loop_cem" else raw_cost
        _, idx = torch.topk(elite_cost, k=topk, dim=1, largest=False)
        batch = torch.arange(num_envs, device=device)[:, None]
        elite = candidates[batch, idx]
        mean = elite.mean(dim=1)
        var = elite.std(dim=1)
        final_cand = candidates.detach().cpu()
        final_raw = raw_cost.detach().cpu()
        final_util = util_score.detach().cpu()
        final_plan = plan_cost.detach().cpu()

    if mode == "raw_cem":
        order_score = -final_raw
    elif mode in {"final_only_rerank", "planner_in_loop_cem"}:
        order_score = -final_plan
    else:
        raise ValueError(mode)
    order = torch.argsort(order_score, dim=1, descending=True)[:, :topk]
    batch_cpu = torch.arange(num_envs)[:, None]
    return {
        "candidates": final_cand[batch_cpu, order],
        "raw_cost": final_raw[batch_cpu, order],
        "utility": final_util[batch_cpu, order],
        "plan_cost": final_plan[batch_cpu, order],
        "score": order_score[batch_cpu, order],
    }


def summarize_labels(labels: np.ndarray, scores: np.ndarray, cands: torch.Tensor, row_base: dict) -> dict:
    first_ranks = []
    aucs = []
    success_over_rank0 = 0
    rank0_failure_with_success = 0
    for lab, score in zip(labels, scores):
        hits = np.nonzero(lab)[0]
        first_ranks.append(int(hits[0] + 1) if len(hits) else None)
        auc = row_auc(score, lab)
        if auc is not None:
            aucs.append(auc)
        if (not lab[0]) and lab.any():
            rank0_failure_with_success += 1
            if score[lab].max() > score[0]:
                success_over_rank0 += 1
    out = dict(row_base)
    topk = labels.shape[1]
    out.update(
        {
            "top1_success": float(labels[:, 0].mean() * 100.0),
            "top3_success": float(labels[:, : min(3, topk)].any(axis=1).mean() * 100.0),
            "top5_success": float(labels[:, : min(5, topk)].any(axis=1).mean() * 100.0),
            "top10_success": float(labels[:, : min(10, topk)].any(axis=1).mean() * 100.0),
            "oracle": float(labels.any(axis=1).mean() * 100.0),
            "near_miss_count": int(((~labels[:, 0]) & labels.any(axis=1)).sum()),
            "success_over_rank0": int(success_over_rank0),
            "rank0_failure_with_success": int(rank0_failure_with_success),
            "success_over_rank0_rate": float(success_over_rank0 / rank0_failure_with_success * 100.0)
            if rank0_failure_with_success
            else 0.0,
            "intra_episode_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "intra_episode_auc_median": float(np.median(aucs)) if aucs else float("nan"),
            "candidate_diversity": candidate_diversity(cands),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
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


def run_split(split: str, train_seeds: list[int], args) -> tuple[list[dict], list[dict]]:
    util = cal.fit_utility(train_seeds)
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
    cases = []
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    for mode in ["raw_cem", "final_only_rerank", "planner_in_loop_cem"]:
        use_lambdas = [0.0] if mode == "raw_cem" else lambdas
        for lamb in use_lambdas:
            out = generate_candidates(
                model,
                prepared_base,
                action_dim=action_dim,
                horizon=int(cfg.plan_config.horizon),
                num_samples=args.num_samples,
                topk=args.topk,
                n_steps=args.cem_steps,
                seed=args.seed,
                util=util,
                lamb=lamb,
                mode=mode,
            )
            labels = eval_topk_plans_batched(cfg, dataset, process, indices, out["candidates"].numpy())
            score_np = out["score"].numpy()
            row = summarize_labels(
                labels,
                score_np,
                out["candidates"],
                {
                    "split": split,
                    "mode": mode,
                    "lambda": float(lamb),
                    "num_eval": args.num_eval,
                    "num_samples": args.num_samples,
                    "cem_steps": args.cem_steps,
                    "topk": args.topk,
                    "mean_raw_cost_top1": float(out["raw_cost"][:, 0].mean()),
                    "mean_plan_cost_top1": float(out["plan_cost"][:, 0].mean()),
                    "mean_utility_top1": float(out["utility"][:, 0].mean()),
                },
            )
            rows.append(row)
            for i, lab in enumerate(labels):
                if (not lab[0]) and lab.any():
                    hits = np.nonzero(lab)[0]
                    cases.append(
                        {
                            "split": split,
                            "mode": mode,
                            "lambda": float(lamb),
                            "eval_i": int(i),
                            "dataset_index": int(indices[i]),
                            "first_success_rank": int(hits[0] + 1),
                            "top1_raw_cost": float(out["raw_cost"][i, 0]),
                            "top1_plan_cost": float(out["plan_cost"][i, 0]),
                            "top1_utility": float(out["utility"][i, 0]),
                        }
                    )
    del model
    torch.cuda.empty_cache()
    return rows, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-eval", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--cem-steps", type=int, default=15)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambdas", default="0.1,0.2,0.5,1.0,2.0,5.0")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_cases = [], []
    for split, (train_seeds, _val_seeds) in SPLITS.items():
        rows, cases = run_split(split, train_seeds, args)
        all_rows.extend(rows)
        all_cases.extend(cases)
        write_csv(out / "moda_only_planner_in_loop_calibrated_cem.csv", all_rows)
        write_csv(out / "moda_only_planner_in_loop_cases.csv", all_cases)

    agg = []
    keys = sorted({(r["mode"], r["lambda"]) for r in all_rows})
    for mode, lamb in keys:
        rs = [r for r in all_rows if r["mode"] == mode and abs(r["lambda"] - lamb) < 1e-9]
        agg.append(
            {
                "mode": mode,
                "lambda": lamb,
                "top1_success": float(np.mean([r["top1_success"] for r in rs])),
                "top3_success": float(np.mean([r["top3_success"] for r in rs])),
                "top5_success": float(np.mean([r["top5_success"] for r in rs])),
                "top10_success": float(np.mean([r["top10_success"] for r in rs])),
                "oracle": float(np.mean([r["oracle"] for r in rs])),
                "near_miss_count": int(sum(r["near_miss_count"] for r in rs)),
                "success_over_rank0_rate": float(np.mean([r["success_over_rank0_rate"] for r in rs])),
                "intra_episode_auc_mean": float(np.nanmean([r["intra_episode_auc_mean"] for r in rs])),
                "candidate_diversity": float(np.mean([r["candidate_diversity"] for r in rs])),
            }
        )
    best = max(agg, key=lambda r: (r["top1_success"], r["oracle"], -r["near_miss_count"]))
    write_csv(out / "moda_only_planner_in_loop_aggregate.csv", agg)
    (out / "moda_only_planner_in_loop_calibrated_cem.json").write_text(
        json.dumps({"settings": vars(args), "rows": all_rows, "aggregate": agg, "best": best}, indent=2) + "\n"
    )
    md = [
        "# MoDA-Only Planner-in-the-Loop Calibrated CEM",
        "",
        "No bsl is used. Utility is trained only from MoDA/stateroll candidate labels. Planner-in-loop CEM uses `J_plan = J_raw - lambda * U_theta(candidate)` for elite selection and final selection.",
        "",
        "|mode|lambda|top1|top3|top5|top10|oracle|near-miss|success>rank0 %|intra AUC|diversity|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        md.append(
            f"|{r['mode']}|{r['lambda']}|{r['top1_success']:.2f}|{r['top3_success']:.2f}|{r['top5_success']:.2f}|"
            f"{r['top10_success']:.2f}|{r['oracle']:.2f}|{r['near_miss_count']}|{r['success_over_rank0_rate']:.2f}|"
            f"{r['intra_episode_auc_mean']:.3f}|{r['candidate_diversity']:.4f}|"
        )
    md += [
        "",
        "## Verdict",
        "",
        f"Best MoDA-only top1 is {best['top1_success']:.2f} with mode={best['mode']} lambda={best['lambda']}.",
    ]
    (out / "moda_only_planner_in_loop_calibrated_cem.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_planner_in_loop_calibrated_cem.md").read_text())


if __name__ == "__main__":
    main()
