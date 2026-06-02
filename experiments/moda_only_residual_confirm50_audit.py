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
import moda_only_learned_residual_proposal as lrp
from moda_only_search_scaling import eval_topk_plans_batched
from topk_oracle_pilot import get_multistart_topk_candidates


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "moda_only_learned_residual_proposal_20260530" / "confirm50_audit"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def labels_metrics(labels: np.ndarray) -> dict:
    return {
        "top1_success": float(labels[:, 0].mean() * 100),
        "top3_success": float(labels[:, : min(3, labels.shape[1])].any(axis=1).mean() * 100),
        "top5_success": float(labels[:, : min(5, labels.shape[1])].any(axis=1).mean() * 100),
        "oracle": float(labels.any(axis=1).mean() * 100),
        "success_density": float(labels.mean() * 100),
        "near_miss_count": int(((~labels[:, 0]) & labels.any(axis=1)).sum()),
    }


def paired_metrics(raw: np.ndarray, other: np.ndarray) -> dict:
    raw_top1 = raw[:, 0].astype(bool)
    other_top1 = other[:, 0].astype(bool)
    return {
        "fixed_vs_raw": int((~raw_top1 & other_top1).sum()),
        "harmed_vs_raw": int((raw_top1 & ~other_top1).sum()),
        "net_vs_raw": int(other_top1.sum() - raw_top1.sum()),
        "both_success": int((raw_top1 & other_top1).sum()),
        "both_fail": int((~raw_top1 & ~other_top1).sum()),
    }


def subset_rows(split: str, subset: str, method: str, scale: float, labels: np.ndarray, raw_labels: np.ndarray | None) -> dict:
    row = {"split": split, "subset": subset, "method": method, "scale": scale}
    row.update(labels_metrics(labels))
    if raw_labels is not None and method != "raw_moda":
        row.update(paired_metrics(raw_labels, labels))
    return row


def choose_indices(cfg, dataset, num_eval: int, seed: int) -> np.ndarray:
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
    return valid_indices[np.sort(rng.choice(len(valid_indices) - 1, size=num_eval, replace=False))]


def eval_subset_rows(
    split: str,
    subset: str,
    subset_idx: np.ndarray,
    raw_labels: np.ndarray,
    method_labels: dict[tuple[str, float], np.ndarray],
) -> list[dict]:
    rows = [subset_rows(split, subset, "raw_moda", 0.0, raw_labels[subset_idx], None)]
    for (method, scale), labels in sorted(method_labels.items()):
        rows.append(subset_rows(split, subset, method, scale, labels[subset_idx], raw_labels[subset_idx]))
    return rows


def run_index_set(
    split: str,
    train_seeds: list[int],
    label: str,
    indices: np.ndarray,
    args,
    scales: list[float],
    subsets: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    x, y = lrp.train_data(train_seeds)
    residual = lrp.fit_ridge(x, y, ridge=args.ridge)
    util = cal.fit_utility(train_seeds)

    cfg = OmegaConf.load("./config/eval/pusht.yaml")
    cfg.policy = lrp.POLICY
    cfg.eval.num_eval = int(len(indices))
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)

    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared_base = base.make_eval_like_info(raw_info, {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)}, process)
    world_tmp = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    low = np.asarray(world_tmp.envs.action_space.low)
    if low.ndim > 1:
        low = low[0]
    action_dim = int(np.prod(low.shape)) * int(cfg.plan_config.action_block)
    wm = base.load_model(cfg, cache_dir=None)
    raw_topk, _raw_costs = get_multistart_topk_candidates(
        wm,
        prepared_base,
        action_dim,
        int(cfg.plan_config.horizon),
        args.num_samples,
        args.raw_topk,
        args.cem_steps,
        args.seed,
        args.restarts,
    )
    raw_labels = eval_topk_plans_batched(cfg, dataset, process, indices, raw_topk[:, : args.eval_topk].numpy())

    prepared = {k: v.clone() if torch.is_tensor(v) else v for k, v in prepared_base.items()}
    prepared = base.expand_info_for_candidates(prepared, raw_topk.shape[0], raw_topk.shape[1])
    raw2, pred, goal = cal.model_rollout_cost(wm, prepared, raw_topk.cuda())
    feat = lrp.online_feature(raw2, pred, goal, raw_topk.cuda())[:, 0]
    delta_small = lrp.predict_ridge(residual, feat).reshape(len(indices), 4, 10)
    delta = np.zeros((len(indices), *raw_topk.shape[2:]), dtype=np.float32)
    h = min(delta.shape[1], delta_small.shape[1])
    d = min(delta.shape[2], delta_small.shape[2])
    delta[:, :h, :d] = delta_small[:, :h, :d]
    delta_t = torch.tensor(delta, dtype=raw_topk.dtype)

    method_labels: dict[tuple[str, float], np.ndarray] = {}
    for scale in scales:
        shifted = raw_topk[:, : args.base_top] + float(scale) * delta_t[:, None]
        pool = torch.cat([raw_topk, shifted], dim=1).cuda()
        raw_s, plan_s, _u = lrp.score_candidates(wm, prepared_base, pool, util, args.cal_lambda)
        for method, score in [("residual_raw_cost", -raw_s), ("residual_calibrated_cost", -plan_s)]:
            order = torch.argsort(score, dim=1, descending=True)[:, : args.eval_topk]
            batch = torch.arange(pool.shape[0], device=pool.device)[:, None]
            selected = pool[batch, order].detach().cpu()
            method_labels[(method, scale)] = eval_topk_plans_batched(cfg, dataset, process, indices, selected.numpy())

    if subsets is None:
        subsets = {label: np.arange(len(indices))}
    metric_rows = []
    for subset_name, subset_idx in subsets.items():
        metric_rows.extend(eval_subset_rows(split, subset_name, subset_idx, raw_labels, method_labels))
    case_rows = []
    best_key = ("residual_calibrated_cost", args.case_scale)
    best_labels = method_labels.get(best_key)
    if best_labels is not None:
        raw_top1 = raw_labels[:, 0].astype(bool)
        best_top1 = best_labels[:, 0].astype(bool)
        for pos, idx in enumerate(indices.tolist()):
            if (not raw_top1[pos]) and best_top1[pos]:
                bucket = "raw_fail_residual_success"
            elif raw_top1[pos] and (not best_top1[pos]):
                bucket = "raw_success_residual_fail"
            elif raw_top1[pos] and best_top1[pos]:
                bucket = "both_success"
            else:
                bucket = "both_fail"
            case_rows.append(
                {
                    "split": split,
                    "index_set": label,
                    "position": pos,
                    "eval_index": int(idx),
                    "bucket": bucket,
                    "raw_top1": int(raw_top1[pos]),
                    "residual_top1": int(best_top1[pos]),
                    "raw_oracle": int(raw_labels[pos].any()),
                    "residual_oracle": int(best_labels[pos].any()),
                    "delta_norm": float(np.linalg.norm(delta[pos])),
                }
            )

    del wm
    torch.cuda.empty_cache()
    return metric_rows, case_rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = sorted({(r["index_set"], r["subset"], r["method"], r["scale"]) for r in rows})
    out = []
    for index_set, subset, method, scale in groups:
        rs = [r for r in rows if (r["index_set"], r["subset"], r["method"], r["scale"]) == (index_set, subset, method, scale)]
        row = {"index_set": index_set, "subset": subset, "method": method, "scale": scale}
        for key in ["top1_success", "top3_success", "top5_success", "oracle", "success_density"]:
            row[key] = float(np.mean([r[key] for r in rs]))
        row["near_miss_count"] = int(sum(r["near_miss_count"] for r in rs))
        for key in ["fixed_vs_raw", "harmed_vs_raw", "net_vs_raw", "both_success", "both_fail"]:
            if key in rs[0]:
                row[key] = int(sum(r.get(key, 0) for r in rs))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--cem-steps", type=int, default=15)
    ap.add_argument("--raw-topk", type=int, default=10)
    ap.add_argument("--eval-topk", type=int, default=10)
    ap.add_argument("--base-top", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--scales", default="0.1,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--case-scale", type=float, default=0.25)
    ap.add_argument("--cal-lambda", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    cfg0 = OmegaConf.load("./config/eval/pusht.yaml")
    cfg0.policy = lrp.POLICY
    dataset0 = base.get_dataset(cfg0)
    medium20 = choose_indices(cfg0, dataset0, 20, args.seed)
    confirm50 = choose_indices(cfg0, dataset0, 50, args.seed)
    index_rows = []
    for name, arr in [("medium20", medium20), ("confirm50", confirm50)]:
        for pos, idx in enumerate(arr.tolist()):
            index_rows.append({"index_set": name, "position": pos, "eval_index": int(idx)})
    write_csv(out / "residual_eval_indices.csv", index_rows)

    rows: list[dict] = []
    case_rows: list[dict] = []
    for split, (train, _val) in lrp.SPLITS.items():
        metric, cases = run_index_set(split, train, "medium20", medium20, args, [args.case_scale])
        rows.extend({"index_set": "medium20", **r} for r in metric)
        case_rows.extend(cases)

        confirm_subsets = {
            "first20": np.arange(0, min(20, len(confirm50))),
            "added30": np.arange(min(20, len(confirm50)), len(confirm50)),
            "all50": np.arange(len(confirm50)),
        }
        metric, cases = run_index_set(split, train, "confirm50_all50", confirm50, args, scales, subsets=confirm_subsets)
        rows.extend({"index_set": "confirm50", **r} for r in metric)
        case_rows.extend(cases)

    agg = aggregate(rows)
    write_csv(out / "moda_only_residual_confirm50_audit.csv", agg)
    write_csv(out / "residual_case_studies.csv", case_rows)

    scale_rows = [r for r in agg if r["index_set"] == "confirm50" and r["subset"] == "all50"]
    write_csv(out / "residual_scale_sensitivity.csv", scale_rows)

    payload = {"settings": vars(args), "indices": index_rows, "rows": rows, "aggregate": agg}
    (out / "moda_only_residual_confirm50_audit.json").write_text(json.dumps(payload, indent=2) + "\n")

    md = [
        "# MoDA-Only Residual Confirm50 Audit",
        "",
        "No bsl, no selector-v3, no risk-controlled integration. This is a paired audit of learned residual proposal only.",
        "",
        "## Aggregate Metrics",
        "",
        "|index_set|method|scale|top1|top3|top5|oracle|near-miss|fixed|harmed|net|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        md.append(
            f"|{r['index_set']}|{r['method']}|{r['scale']}|{r['top1_success']:.2f}|"
            f"{r['top3_success']:.2f}|{r['top5_success']:.2f}|{r['oracle']:.2f}|"
            f"{r['near_miss_count']}|{r.get('fixed_vs_raw', '')}|{r.get('harmed_vs_raw', '')}|{r.get('net_vs_raw', '')}|"
        )
    md += [
        "",
        "## Files",
        "",
        "- `residual_eval_indices.csv`",
        "- `moda_only_residual_confirm50_audit.csv`",
        "- `residual_scale_sensitivity.csv`",
        "- `residual_case_studies.csv`",
    ]
    (out / "moda_only_residual_confirm50_audit.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_residual_confirm50_audit.md").read_text())


if __name__ == "__main__":
    main()
