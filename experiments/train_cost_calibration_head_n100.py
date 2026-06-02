from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "cost_calibration_head_n100_20260529"
BSL_ACTION = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
BSL_RAW = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
GRID = ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def load_src(action_dir: Path, raw_dir: Path, variant: str, seed: int) -> dict:
    a = np.load(action_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(raw_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {
        "actions": a["actions"].astype(np.float64),
        "costs": a["costs"].astype(np.float64),
        "labels": a["labels"].astype(bool),
        "pred": r["pred"].astype(np.float64),
        "goal": r["goal"].astype(np.float64),
        "indices": a["indices"],
    }


def load_all() -> dict[int, dict]:
    data = {}
    for seed in SEEDS:
        b = load_src(BSL_ACTION, BSL_RAW, "baseline", seed)
        st = load_src(ST_ACTION, ST_RAW, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        data[seed] = {"b": b, "st": st}
    return data


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


def traj_stats(pred: np.ndarray, goal: np.ndarray) -> dict:
    g = goal_for_pred(goal, pred)
    dist = np.sqrt(((pred - g) ** 2).sum(axis=-1))
    return {
        "final": dist[:, :, -1],
        "mean": dist.mean(axis=2),
        "min": dist.min(axis=2),
        "progress": dist[:, :, 0] - dist[:, :, -1],
        "latent_mean": pred.mean(axis=(2, 3)),
        "latent_std": pred.std(axis=(2, 3)),
    }


def action_stats(actions: np.ndarray) -> dict:
    norm_t = np.sqrt((actions**2).sum(axis=-1))
    return {
        "norm": norm_t.mean(axis=2),
        "std": norm_t.std(axis=2),
        "final_norm": norm_t[:, :, -1],
    }


def entropy_from_cost(costs: np.ndarray) -> np.ndarray:
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    sum_pos = ranks[y].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def zscore_row(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def build_episode_features(data: dict, seed: int, ep: int) -> dict:
    b = data[seed]["b"]
    st = data[seed]["st"]
    bc = b["costs"][ep]
    sc = st["costs"][ep]
    bs = np.sort(bc)
    ss = np.sort(sc)
    btr = traj_stats(b["pred"][ep : ep + 1], b["goal"][ep : ep + 1])
    strj = traj_stats(st["pred"][ep : ep + 1], st["goal"][ep : ep + 1])
    st_best = int(np.argmin(sc))
    return {
        "bsl_margin_top2": float(bs[1] - bs[0]),
        "bsl_margin_top5": float(bs[min(4, len(bs) - 1)] - bs[0]),
        "bsl_margin_top10": float(bs[min(9, len(bs) - 1)] - bs[0]),
        "bsl_cost_std": float(bc.std()),
        "bsl_cost_entropy": float(entropy_from_cost(bc[None])[0]),
        "bsl_top1_cost": float(bc[0]),
        "st_minus_bsl_best_cost": float(ss[0] - bs[0]),
        "bsl_rank0_minus_st_best_final_dist": float(btr["final"][0, 0] - strj["final"][0, st_best]),
        "bsl_rank0_minus_st_best_mean_dist": float(btr["mean"][0, 0] - strj["mean"][0, st_best]),
        "st_best_minus_bsl_rank0_progress": float(strj["progress"][0, st_best] - btr["progress"][0, 0]),
    }


def candidate_matrix(data: dict, seeds: list[int]):
    X, y, meta, raw_score = [], [], [], []
    for seed in seeds:
        b = data[seed]["b"]
        st = data[seed]["st"]
        btr = traj_stats(b["pred"], b["goal"])
        strj = traj_stats(st["pred"], st["goal"])
        bact = action_stats(b["actions"])
        stact = action_stats(st["actions"])
        for ep in range(b["labels"].shape[0]):
            ep_feat = build_episode_features(data, seed, ep)
            bcost = b["costs"][ep]
            scost = st["costs"][ep]
            union_cost = np.concatenate([bcost, scost])
            union_z = zscore_row(union_cost)
            b_z = zscore_row(bcost)
            st_z = zscore_row(scost)
            bsl_success = bool(b["labels"][ep, 0])
            bsl_oracle = bool(b["labels"][ep].any())
            st_oracle = bool(st["labels"][ep].any())
            st_only = (not bsl_success) and st_oracle and (not bsl_oracle)

            for source, arr, labels, tr, act, offset in [
                ("bsl", b, b["labels"][ep], btr, bact, 0),
                ("stateroll", st, st["labels"][ep], strj, stact, 30),
            ]:
                costs = bcost if source == "bsl" else scost
                local_z = b_z if source == "bsl" else st_z
                for j in range(costs.shape[0]):
                    uj = offset + j
                    feat = [
                        1.0 if source == "stateroll" else 0.0,
                        float(j) / max(1, costs.shape[0] - 1),
                        float(costs[j]),
                        float(local_z[j]),
                        float(union_z[uj]),
                        float(costs[j] - costs.min()),
                        float(costs[j] - bcost[0]),
                        float(tr["final"][ep, j]),
                        float(tr["mean"][ep, j]),
                        float(tr["min"][ep, j]),
                        float(tr["progress"][ep, j]),
                        float(act["norm"][ep, j]),
                        float(act["std"][ep, j]),
                        float(tr["latent_mean"][ep, j]),
                        float(tr["latent_std"][ep, j]),
                        ep_feat["bsl_margin_top2"],
                        ep_feat["bsl_margin_top5"],
                        ep_feat["bsl_margin_top10"],
                        ep_feat["bsl_cost_std"],
                        ep_feat["bsl_cost_entropy"],
                        ep_feat["st_minus_bsl_best_cost"],
                        ep_feat["bsl_rank0_minus_st_best_final_dist"],
                        ep_feat["bsl_rank0_minus_st_best_mean_dist"],
                        ep_feat["st_best_minus_bsl_rank0_progress"],
                        1.0 if bsl_success else 0.0,
                        1.0 if st_only else 0.0,
                    ]
                    X.append(feat)
                    y.append(bool(labels[j]))
                    raw_score.append(float(-costs[j]))
                    meta.append(
                        {
                            "seed": seed,
                            "episode": ep,
                            "source": source,
                            "local_rank": j,
                            "union_index": uj,
                            "bsl_success": bsl_success,
                            "bsl_oracle": bsl_oracle,
                            "st_oracle": st_oracle,
                            "stateroll_only_fixable": st_only,
                        }
                    )
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=bool), meta, np.asarray(raw_score, dtype=np.float64)


def fit_logistic(X: np.ndarray, y: np.ndarray, weights: np.ndarray, lr=0.03, epochs=2500, l2=1e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    yv = y.astype(np.float64)
    weights = weights.astype(np.float64)
    weights = weights / (weights.mean() + 1e-12)
    for _ in range(epochs):
        z = np.clip(Xb @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xb.T @ ((p - yv) * weights)) / Xb.shape[0]
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    z = np.clip(Xb @ model["w"], -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def rank_metrics(scores_by_ep: dict, labels_by_ep: dict, max_k: int):
    first_ranks = []
    no_success = 0
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 30: 0}
    near_miss = 0
    for key, scores in scores_by_ep.items():
        labels = labels_by_ep[key]
        order = np.argsort(-scores, kind="stable")
        if not labels.any():
            no_success += 1
            continue
        pos = np.where(labels[order])[0]
        first_ranks.append(int(pos[0] + 1))
        if not labels[0]:
            near_miss += 1
        for k in top_hits:
            kk = min(k, max_k)
            if labels[order[:kk]].any():
                top_hits[k] += 1
    n = len(scores_by_ep)
    arr = np.asarray(first_ranks, dtype=np.float64)
    return {
        "episodes": n,
        "episodes_with_success": int(n - no_success),
        "episodes_without_success": int(no_success),
        "first_success_rank_mean": float(arr.mean()) if arr.size else None,
        "first_success_rank_median": float(np.median(arr)) if arr.size else None,
        "near_miss_failure_count": int(near_miss),
        **{f"top{k}_success_recall": float(top_hits[k] / n * 100.0) for k in top_hits},
    }


def evaluate_candidate_scores(X, y, meta, scores, raw_scores, seeds):
    rows = []
    for pool in ["bsl", "stateroll", "union"]:
        mask_seed = np.asarray([m["seed"] in seeds for m in meta], dtype=bool)
        if pool != "union":
            mask_pool = np.asarray([m["source"] == pool for m in meta], dtype=bool)
        else:
            mask_pool = np.ones(len(meta), dtype=bool)
        mask = mask_seed & mask_pool
        auc_cal = binary_auc(y[mask].astype(int), scores[mask])
        auc_raw = binary_auc(y[mask].astype(int), raw_scores[mask])

        scores_by_ep, raw_by_ep, labels_by_ep = {}, {}, {}
        for i in np.where(mask)[0]:
            m = meta[i]
            key = (m["seed"], m["episode"])
            scores_by_ep.setdefault(key, []).append(scores[i])
            raw_by_ep.setdefault(key, []).append(raw_scores[i])
            labels_by_ep.setdefault(key, []).append(y[i])
        scores_by_ep = {k: np.asarray(v, dtype=np.float64) for k, v in scores_by_ep.items()}
        raw_by_ep = {k: np.asarray(v, dtype=np.float64) for k, v in raw_by_ep.items()}
        labels_by_ep = {k: np.asarray(v, dtype=bool) for k, v in labels_by_ep.items()}
        rm_cal = rank_metrics(scores_by_ep, labels_by_ep, 60 if pool == "union" else 30)
        rm_raw = rank_metrics(raw_by_ep, labels_by_ep, 60 if pool == "union" else 30)
        row = {"pool": pool, "candidate_auc_raw": auc_raw, "candidate_auc_calibrated": auc_cal}
        for k, v in rm_raw.items():
            row[f"raw_{k}"] = v
        for k, v in rm_cal.items():
            row[f"cal_{k}"] = v
        rows.append(row)
    return rows


def choose_threshold(meta, y, scores, train_seeds):
    # Threshold on best stateroll calibrated score. Train-only labels decide the threshold.
    by_ep = defaultdict_episode(meta, y, scores, train_seeds)
    max_scores = np.asarray([v["best_score"] for v in by_ep.values()], dtype=np.float64)
    thresholds = np.unique(np.quantile(max_scores, np.linspace(0, 0.95, 20)).round(6))
    rows = []
    for thr in thresholds:
        fixed = harmed = switches = 0
        for v in by_ep.values():
            if v["best_score"] <= thr:
                continue
            switches += 1
            succ = v["best_success"]
            if (not v["bsl_success"]) and succ:
                fixed += 1
            if v["bsl_success"] and (not succ):
                harmed += 1
        rows.append({"threshold": float(thr), "fixed": fixed, "harmed": harmed, "switches": switches})
    safe = [r for r in rows if r["harmed"] <= 1 and r["switches"] <= 20]
    cand = safe if safe else rows
    cand.sort(key=lambda r: (r["fixed"] - 3 * r["harmed"], -r["harmed"], -r["switches"], r["threshold"]), reverse=True)
    return cand[0], rows


def defaultdict_episode(meta, y, scores, seeds):
    out = {}
    for i, m in enumerate(meta):
        if m["seed"] not in seeds or m["source"] != "stateroll":
            continue
        key = (m["seed"], m["episode"])
        v = out.setdefault(
            key,
            {
                "best_score": -1e30,
                "best_success": False,
                "best_rank": -1,
                "bsl_success": m["bsl_success"],
                "stateroll_only_fixable": m["stateroll_only_fixable"],
            },
        )
        if scores[i] > v["best_score"]:
            v["best_score"] = float(scores[i])
            v["best_success"] = bool(y[i])
            v["best_rank"] = int(m["local_rank"])
    return out


def fixed_gate_selected(split: str):
    grid = json.loads(GRID.read_text())["records"]
    if split.startswith("splitA"):
        target = ("extratrees", "top10+st_gap_bottom20+AND")
    else:
        target = ("randomforest", "top10+abs_gap_top10+AND")
    for r in grid:
        if r["split"] == split and r["model"] == target[0] and r["rule"] == target[1]:
            return {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}, r
    raise RuntimeError(f"missing fixed gate row for {split}")


def eval_final_selection(split: str, meta, y, scores, val_seeds, threshold):
    gate_set, gate_row = fixed_gate_selected(split)
    by_ep = defaultdict_episode(meta, y, scores, val_seeds)
    fixed = harmed = switches = st_only_recovered = 0
    bsl_success_count = 0
    total_eps = 0
    selected_rows = []
    for key, v in by_ep.items():
        total_eps += 1
        if v["bsl_success"]:
            bsl_success_count += 1
        if key not in gate_set:
            continue
        if v["best_score"] <= threshold:
            continue
        switches += 1
        succ = v["best_success"]
        if (not v["bsl_success"]) and succ:
            fixed += 1
        if v["bsl_success"] and (not succ):
            harmed += 1
        if v["stateroll_only_fixable"] and succ:
            st_only_recovered += 1
        selected_rows.append(
            {
                "seed": key[0],
                "episode": key[1],
                "best_score": v["best_score"],
                "best_rank": v["best_rank"],
                "best_success": succ,
                "bsl_success": v["bsl_success"],
                "stateroll_only_fixable": v["stateroll_only_fixable"],
            }
        )
    return {
        "split": split,
        "gate_rule": gate_row["rule"],
        "gate_model": gate_row["model"],
        "gate_selected": len(gate_set),
        "gate_st_only": int(gate_row["stateroll_only_captured"]),
        "gate_bsl_fp": int(gate_row["bsl_success_false_positive"]),
        "threshold": float(threshold),
        "episodes": int(total_eps),
        "bsl_top1": float(bsl_success_count / total_eps * 100.0) if total_eps else 0.0,
        "calibrated_selector_top1": float((bsl_success_count + fixed - harmed) / total_eps * 100.0) if total_eps else 0.0,
        "fixed": int(fixed),
        "harmed": int(harmed),
        "switches": int(switches),
        "stateroll_only_recovered": int(st_only_recovered),
        "selected_rows": selected_rows,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], list):
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    X, y, meta, raw_scores = candidate_matrix(data, SEEDS)
    records = []
    all_final = []
    all_threshold_rows = []
    for split, (train_seeds, val_seeds) in SPLITS.items():
        train_mask = np.asarray([m["seed"] in train_seeds for m in meta], dtype=bool)
        # Class balance plus more weight for stateroll-only fixable positives.
        pos = y[train_mask].sum()
        neg = train_mask.sum() - pos
        weights = np.ones(train_mask.sum(), dtype=np.float64)
        weights[y[train_mask]] = max(1.0, neg / max(1, pos))
        train_meta = [m for i, m in enumerate(meta) if train_mask[i]]
        for i, m in enumerate(train_meta):
            if m["source"] == "stateroll" and m["stateroll_only_fixable"] and y[np.where(train_mask)[0][i]]:
                weights[i] *= 3.0
        model = fit_logistic(X[train_mask], y[train_mask], weights)
        cal_scores = predict(model, X)
        train_metrics = evaluate_candidate_scores(X, y, meta, cal_scores, raw_scores, train_seeds)
        val_metrics = evaluate_candidate_scores(X, y, meta, cal_scores, raw_scores, val_seeds)
        thr_best, thr_rows = choose_threshold(meta, y, cal_scores, train_seeds)
        final = eval_final_selection(split, meta, y, cal_scores, val_seeds, thr_best["threshold"])
        final["train_threshold_fixed"] = thr_best["fixed"]
        final["train_threshold_harmed"] = thr_best["harmed"]
        final["train_threshold_switches"] = thr_best["switches"]
        records.append(
            {
                "split": split,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "threshold_choice": thr_best,
                "final": final,
            }
        )
        for row in val_metrics:
            records_row = {"split": split, "stage": "val", **row}
            all_final.append(records_row)
        all_threshold_rows.extend([{"split": split, **r} for r in thr_rows])

    summary = {"records": records}
    (OUT / "cost_calibration_head_n100.json").write_text(json.dumps(summary, indent=2))
    write_csv(OUT / "cost_calibration_head_n100.csv", all_final)
    write_csv(OUT / "cost_calibration_threshold_grid_n100.csv", all_threshold_rows)

    lines = ["# MoDA Cost Calibration Head n100", ""]
    lines.append("Frozen candidate pools. No encoder/predictor/candidate-pool changes. A small linear calibration head is trained on candidate features only.")
    lines.append("")
    lines.append("## Main Table: Raw Stateroll Cost vs Calibrated Stateroll Score")
    lines.append("")
    lines.append("|split|score|candidate AUC|first-success rank mean|first-success rank median|top1|top3|top5|top10|top30|near-miss|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        split = rec["split"]
        for row in rec["val_metrics"]:
            if row["pool"] != "stateroll":
                continue
            lines.append(
                f"|{split}|raw stateroll cost|{row['candidate_auc_raw']:.3f}|"
                f"{row['raw_first_success_rank_mean']:.2f}|{row['raw_first_success_rank_median']:.2f}|"
                f"{row['raw_top1_success_recall']:.1f}|{row['raw_top3_success_recall']:.1f}|{row['raw_top5_success_recall']:.1f}|"
                f"{row['raw_top10_success_recall']:.1f}|{row['raw_top30_success_recall']:.1f}|{row['raw_near_miss_failure_count']}|"
            )
            lines.append(
                f"|{split}|calibrated stateroll score|{row['candidate_auc_calibrated']:.3f}|"
                f"{row['cal_first_success_rank_mean']:.2f}|{row['cal_first_success_rank_median']:.2f}|"
                f"{row['cal_top1_success_recall']:.1f}|{row['cal_top3_success_recall']:.1f}|{row['cal_top5_success_recall']:.1f}|"
                f"{row['cal_top10_success_recall']:.1f}|{row['cal_top30_success_recall']:.1f}|{row['cal_near_miss_failure_count']}|"
            )
    lines.append("")
    lines.append("## Candidate-Level Validation Metrics: All Pools")
    lines.append("")
    lines.append("|split|pool|raw AUC|cal AUC|raw first rank|cal first rank|raw top1|cal top1|raw top10|cal top10|raw near-miss|cal near-miss|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        split = rec["split"]
        for row in rec["val_metrics"]:
            lines.append(
                f"|{split}|{row['pool']}|{row['candidate_auc_raw']:.3f}|{row['candidate_auc_calibrated']:.3f}|"
                f"{row['raw_first_success_rank_mean']:.2f}|{row['cal_first_success_rank_mean']:.2f}|"
                f"{row['raw_top1_success_recall']:.1f}|{row['cal_top1_success_recall']:.1f}|"
                f"{row['raw_top10_success_recall']:.1f}|{row['cal_top10_success_recall']:.1f}|"
                f"{row['raw_near_miss_failure_count']}|{row['cal_near_miss_failure_count']}|"
            )
    lines.append("")
    lines.append("## Fixed Precision Gate Final Selection")
    lines.append("")
    lines.append("|split|gate|bsl top1|calibrated selector top1|gate selected|gate st-only|gate bsl FP|threshold|fixed|harmed|switches|st-only recovered|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    total_fixed = total_harmed = total_switches = total_st = 0
    for rec in records:
        f = rec["final"]
        total_fixed += f["fixed"]
        total_harmed += f["harmed"]
        total_switches += f["switches"]
        total_st += f["stateroll_only_recovered"]
        lines.append(
            f"|{f['split']}|{f['gate_model']} {f['gate_rule']}|{f['bsl_top1']:.1f}|{f['calibrated_selector_top1']:.1f}|"
            f"{f['gate_selected']}|{f['gate_st_only']}|{f['gate_bsl_fp']}|{f['threshold']:.4f}|"
            f"{f['fixed']}|{f['harmed']}|{f['switches']}|{f['stateroll_only_recovered']}|"
        )
    lines.append("")
    lines.append(f"OOF fixed={total_fixed}, harmed={total_harmed}, switches={total_switches}, stateroll-only recovered={total_st}.")
    (OUT / "cost_calibration_head_n100.md").write_text("\n".join(lines) + "\n")
    print((OUT / "cost_calibration_head_n100.md").read_text())


if __name__ == "__main__":
    main()
